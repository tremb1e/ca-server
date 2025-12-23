import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()

        self.sos_token = args.sos_token
        self.codebook_hw = getattr(args, "codebook_shape", (3, 6))

        self.vqgan = self.load_vqgan(args)

        # 小型 GPT，超参由 args 控制，保持在 5–30M 量级
        target_block = getattr(args, "gpt_block_size", getattr(args, "block_size", 32))
        target_block = max(16, min(32, int(target_block)))  # 遵循 16–32 的要求

        transformer_config = {
            "vocab_size": args.num_codebook_vectors,
            "block_size": target_block,
            "n_layer": getattr(args, "gpt_n_layer", 6),
            "n_head": getattr(args, "gpt_n_head", 6),
            "n_embd": getattr(args, "gpt_n_embd", 384),
            "embd_pdrop": getattr(args, "gpt_embd_pdrop", 0.1),
            "resid_pdrop": getattr(args, "gpt_resid_pdrop", 0.1),
            "attn_pdrop": getattr(args, "gpt_attn_pdrop", 0.1),
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = args.pkeep

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        self.codebook_hw = quant_z.shape[2:]
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def z_to_image(self, indices, p1=None, p2=None):
        h, w = self.codebook_hw if p1 is None or p2 is None else (p1, p2)
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], h, w, self.vqgan.codebook.latent_dim)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x):
        _, indices = self.encode_to_z(x)

        sos_tokens = torch.full((x.shape[0], 1), self.sos_token, device=indices.device, dtype=torch.long)

        # 序列长度超过 block_size 时做等间距下采样，确保符合小 GPT 的上下文长度
        max_tokens = self.transformer.config.block_size - 1  # 预留 SOS
        if indices.shape[1] > max_tokens:
            positions = torch.linspace(0, indices.shape[1] - 1, steps=max_tokens, device=indices.device)
            positions = positions.long()
            indices = torch.gather(indices, 1, positions.expand(indices.size(0), -1))

        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device=indices.device))
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            # 仅保留需要的上下文长度，保证与 block_size 对齐
            if x.size(1) > self.transformer.config.block_size:
                x_input = x[:, -self.transformer.config.block_size :]
            else:
                x_input = x

            logits, _ = self.transformer(x_input)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1]:]
        self.transformer.train()
        return x

    @torch.no_grad()
    def log_images(self, x):
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.full((x.shape[0], 1), self.sos_token, device=indices.device, dtype=torch.long)

        start_indices = indices[:, :indices.shape[1] // 2]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1] - start_indices.shape[1])
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps=indices.shape[1])
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, full_sample, half_sample, torch.concat((x, x_rec, half_sample, full_sample))
















