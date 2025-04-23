from PIL import Image
import numpy as np
from lang_sam import LangSAM
from lang_sam.utils import draw_image

def main():
    model = LangSAM()
    image_pil = Image.open("./assets/car.jpeg").convert("RGB")
    text_prompt = "car. wheel."

    results = model.predict([image_pil], [text_prompt])
    print("results:", results)

    if not results or 'masks' not in results[0] or 'labels' not in results[0]:
        print("❌ マスクまたはラベルが見つかりませんでした。")
        return

    masks = results[0]['masks']
    boxes = results[0]['boxes']
    scores = results[0]['scores']
    labels = results[0]['labels']

    image_np = np.array(image_pil)

    # utils.py の draw_image を使って描画
    annotated_image = draw_image(
        image_rgb=image_np,
        masks=np.array(masks),
        xyxy=np.array(boxes),
        probs=np.array(scores),
        labels=labels
    )

    # 保存
    Image.fromarray(annotated_image).save("masked_image_all.png")
    print("✅ マスク付き画像を保存しました: masked_image_all.png")

if __name__ == "__main__":
    main()
