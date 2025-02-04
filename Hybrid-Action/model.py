import torch
import torch.nn as nn
from transformers import BertForMaskedLM

class BERTPrompt4NR(nn.Module):
    def __init__(self, model_name, answer_ids, args):
        super(BERTPrompt4NR, self).__init__()
        # Load mô hình ở chế độ FP16 và tối ưu bộ nhớ CPU
        self.BERT = BertForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.BERT.resize_token_embeddings(args.vocab_size)

        # Cho phép cập nhật tất cả các tham số
        for param in self.BERT.parameters():
            param.requires_grad = True

        self.answer_ids = answer_ids
        self.mask_token_id = 103
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, batch_enc, batch_attn, batch_labs):
        outputs = self.BERT(input_ids=batch_enc,
                            attention_mask=batch_attn)
        out_logits = outputs.logits

        # Lấy vị trí của [MASK] trong input
        mask_position = batch_enc.eq(self.mask_token_id)
        mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]

        answer_logits = mask_logits[:, self.answer_ids]
        loss = self.loss_func(answer_logits, batch_labs)

        return loss, answer_logits.softmax(dim=1)
