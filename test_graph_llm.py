import json
import pathlib
import pickle
import transformers
import torch
import os
import copy
import time
import gc
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from config import *
from transformers import default_data_collator
from accelerate import DistributedDataParallelKwargs

import os
from utils import *
from llama import Transformer, ModelArgs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

os.environ["WANDB_DISABLED"] = "true"


def main(args, SEED):
    group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.model_name],
                                                "group": group,
                                                "name": f"{args.dataset}_EXP{SEED}",
                                                "config": args}
                                           },
                              )

    seed_everything(seed=SEED)
    accelerator.print(args)


    with accelerator.main_process_first():
        tokenizer = LlamaTokenizer.from_pretrained('Llama-2-7b-hf')
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'

        train_split, config_split, split, edge_index = load_dataset[args.dataset](args.dataset_path)

        original_dataset = config_split.map(
            preprocess_original_dataset[args.dataset](tokenizer=tokenizer, max_length=original_len[args.dataset]),
            batched=True,
            batch_size=None,
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=16,
        ).with_format("torch")

        clm_dataset_train = train_split.map(
            preprocess_train_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=16,
        ).with_format("torch")


        clm_dataset_test = train_split.map(
            preprocess_test_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=16,
        ).with_format("torch")

    accelerator.wait_for_everyone()

    # Step 2: Build Node Classification Dataset
    train_dataset = clm_dataset_train.select(split["train"])
    val_dataset = clm_dataset_train.select(split["valid"])
    val_dataset_eval = clm_dataset_test.select(split["valid"])
    test_dataset = clm_dataset_train.select(split["test"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    val_loader_eval = torch.utils.data.DataLoader(
        val_dataset_eval,
        batch_size=args.batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    with open(Path(f"{module_path}/{args.model_name}/") / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        w_lora=False,
        w_adapter=True,
        adapter_layer=8,
        adapter_dim=args.adapter_dim,
        adapter_len=args.adapter_len,
        lora_alpha=16,
        lora_r=8,
        num_hops=3,
        n_mp_layers=args.n_mp_layers,
        rrwp=args.rrwp,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        adapter_n_heads=args.adapter_n_heads,
        task_level=task_level[args.dataset],
        **params,
    )

    model_args.vocab_size = tokenizer.vocab_size
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    base_model: Transformer = Transformer(
        params=model_args,
        edge_index=edge_index,
        input_ids=original_dataset["input_ids"],
        input_attention_mask=original_dataset["attention_mask"],
    )
    torch.set_default_tensor_type(torch.FloatTensor)

    ckpt = Path(f"{module_path}/{args.model_name}/consolidated.00.pth")
    ckpt = torch.load(ckpt, map_location="cpu")
    base_model.load_state_dict(ckpt, strict=False)

    accelerator.print(model_args)

    # Step 4 Set Optimizer
    param_adapter, param_lora = base_model.set_trainable_params_new()

    lr_group = {
        "adapter": args.lr,
        "lora": args.lr,
    }

    wd_group = {
        "adapter": args.wd,
        "lora": args.wd,
    }

    accelerator.print(lr_group)
    accelerator.print(wd_group)

    optimizer = torch.optim.AdamW(
        [
            {"params": param_adapter, "lr": lr_group["adapter"], "weight_decay": wd_group["adapter"]},
            {"params": param_lora, "lr": lr_group["lora"], "weight_decay": wd_group["lora"]},
        ],
        betas=(0.9, 0.95),
    )

    trainable_params, all_param = base_model.print_trainable_params()
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    base_model.load_state_dict(torch.load(args.save_path).state_dict(), strict=False)
    model, train_loader, val_loader, val_loader_eval, optimizer = accelerator.prepare(
        base_model, train_loader, val_loader, val_loader_eval, optimizer
    )
    # Step 5. Evaluating

    model, test_loader = accelerator.prepare(model, test_loader)

    model.eval()
    val_loss = 0.0
    total_acc = 0

    progress_bar_test = tqdm(range(len(test_loader)))

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            loss, temp_acc = model(**batch)
            val_loss += loss.item()
            total_acc += temp_acc
            progress_bar_test.update()

        accelerator.print(f"Test Loss: {val_loss / len(test_loader)}")
        accelerator.print(f"Test Acc: {total_acc / len(test_loader)}")


if __name__ == "__main__":

    args = parse_args_llama()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(
            log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs], gradient_accumulation_steps=args.grad_steps
        )
        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
