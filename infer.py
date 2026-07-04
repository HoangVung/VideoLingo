import json
from pathlib import Path

import torch


def find_latest_checkpoint(model_dir, prefix="G"):
    model_path = Path(model_dir)
    candidates = []
    candidates.extend(model_path.glob(f"{prefix}_*.pth"))
    candidates.extend(model_path.glob(f"{prefix}.pth"))
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict) and key != "spk2id":
                value = HParams(**value)
            setattr(self, key, value)


class VietnameseTTS:
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        from src.models import SynthesizerTrn
        from src.text.symbols import num_languages, num_tones, symbols

        self.device = device
        self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as file:
            config = json.load(file)

        self.hps = HParams(**config)
        self.sampling_rate = self.hps.data.sampling_rate
        self.spk2id = dict(self.hps.data.spk2id)
        self.speakers = list(self.spk2id.keys())
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        model_num_tones = state_dict.get("enc_p.tone_emb.weight", torch.empty(num_tones)).shape[0]
        model_num_languages = state_dict.get("enc_p.language_emb.weight", torch.empty(num_languages)).shape[0]

        self.model = SynthesizerTrn(
            n_vocab=len(symbols),
            spec_channels=self.hps.data.filter_length // 2 + 1,
            segment_size=self.hps.train.segment_size // self.hps.data.hop_length,
            inter_channels=self.hps.model.inter_channels,
            hidden_channels=self.hps.model.hidden_channels,
            filter_channels=self.hps.model.filter_channels,
            n_heads=self.hps.model.n_heads,
            n_layers=self.hps.model.n_layers,
            kernel_size=self.hps.model.kernel_size,
            p_dropout=self.hps.model.p_dropout,
            resblock=self.hps.model.resblock,
            resblock_kernel_sizes=self.hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hps.model.resblock_dilation_sizes,
            upsample_rates=self.hps.model.upsample_rates,
            upsample_initial_channel=self.hps.model.upsample_initial_channel,
            upsample_kernel_sizes=self.hps.model.upsample_kernel_sizes,
            n_speakers=self.hps.data.n_speakers,
            gin_channels=self.hps.model.gin_channels,
            use_sdp=True,
            n_layers_trans_flow=self.hps.model.n_layers_trans_flow,
            use_transformer_flow=getattr(self.hps.model, "use_transformer_flow", True),
            num_languages=model_num_languages,
            num_tones=model_num_tones,
        ).to(self.device).eval()

        self.model.load_state_dict(state_dict, strict=False)

    def synthesize(
        self,
        text,
        speaker=None,
        length_scale=1.0,
        noise_scale=0.667,
        noise_scale_w=0.8,
        sdp_ratio=0.0,
    ):
        from src.nn import commons
        from src.text import cleaned_text_to_sequence
        from src.vietnamese.phonemizer import text_to_phonemes
        from src.vietnamese.text_processor import process_vietnamese_text

        speaker = speaker or self.speakers[0]
        if speaker not in self.spk2id:
            raise ValueError(f"Unknown Valtec speaker '{speaker}'. Available: {', '.join(self.speakers)}")

        processed = process_vietnamese_text(text)
        phones, tones_raw, _ = text_to_phonemes(processed)
        phone_ids, tone_ids, language_ids = cleaned_text_to_sequence(phones, tones_raw, "VI")

        if getattr(self.hps.data, "add_blank", True):
            phone_ids = commons.intersperse(phone_ids, 0)
            tone_ids = commons.intersperse(tone_ids, 0)
            language_ids = commons.intersperse(language_ids, 0)

        phone_t = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        tone_t = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        lang_t = torch.LongTensor(language_ids).unsqueeze(0).to(self.device)
        phone_len = torch.LongTensor([phone_t.shape[1]]).to(self.device)
        sid = torch.LongTensor([self.spk2id[speaker]]).to(self.device)
        bert = torch.zeros(1, 1024, phone_t.shape[1], device=self.device)
        ja_bert = torch.zeros(1, 768, phone_t.shape[1], device=self.device)

        with torch.no_grad():
            audio, *_ = self.model.infer(
                phone_t,
                phone_len,
                sid=sid,
                tone=tone_t,
                language=lang_t,
                bert=bert,
                ja_bert=ja_bert,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sdp_ratio=sdp_ratio,
            )

        return audio[0, 0].cpu().numpy(), self.sampling_rate
