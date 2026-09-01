import torch
from transformers import CLIPModel, CLIPProcessor
import os

MODEL_NAME = "openai/clip-vit-base-patch32"
TRAIT_PROMPTS = {
 "EarShape": {"FloppyEars":"a dog with floppy ears","ErectEars":"a dog with erect upright ears","SemiErectEars":"a dog with semi-erect ears","ButtonEars":"a dog with small button ears"},
 "CoatType": {"ShortCoat":"a dog with a short coat","MediumCoat":"a dog with a medium-length coat","LongCoat":"a dog with a long coat","WireCoat":"a dog with a wiry coat"},
 "CoatPattern": {"SolidColor":"a dog with a solid single-color coat","Brindle":"a dog with a brindle coat pattern","Spotted":"a dog with a spotted coat pattern","Tricolor":"a dog with a tricolor coat","PatchedColor":"a dog with patches of different coat colors"},
 "SizeClass": {"ToySize":"a toy-sized dog","SmallSize":"a small-sized dog","MediumSize":"a medium-sized dog","LargeSize":"a large-sized dog","GiantSize":"a giant-sized dog"},
 "SnoutLength": {"ShortSnout":"a dog with a short snout","MediumSnout":"a dog with a medium-length snout","LongSnout":"a dog with a long snout"},
 "TailCarriage": {"CurledTail":"a dog with a curled tail","StraightTail":"a dog with a straight tail","PlumeTail":"a dog with a long plume-like tail","DockedTail":"a dog with a docked tail"},
}

def main():
    print("Loading CLIPModel for text processing...")
    clip = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    text_embeds_dict = {}
    with torch.no_grad():
        for t, pm in TRAIT_PROMPTS.items():
            labels = list(pm)
            texts = [pm[x] for x in labels]
            inputs = processor(text=texts, return_tensors="pt", padding=True)
            outputs = clip.text_projection(clip.text_model(**inputs).pooler_output)
            # Normalize text embeddings
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            text_embeds_dict[t] = outputs.cpu()

    # Get logit scale parameter for temperature scaling
    logit_scale = clip.logit_scale.item()

    out_path = os.path.join("hf_space", "trained_model_fact_checked", "text_embeds.pt")
    torch.save({
        "embeds": text_embeds_dict,
        "logit_scale": logit_scale
    }, out_path)
    print(f"Successfully saved text embeddings to {out_path}")

if __name__ == "__main__":
    main()
