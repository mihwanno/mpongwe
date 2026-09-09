# src/provider.py
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional
import io

from pdf2image import convert_from_path
from openai import OpenAI, AsyncOpenAI
import pdb


PROMPT_DIR = Path(__file__).parent / "prompts"


def pil_to_base64(image_object, format='JPEG', quality=70):
    """
    Converts a PIL Image object to a base64 encoded string.

    Args:
        image_object: The PIL Image object.
        format: The image format for compression (e.g., 'JPEG', 'PNG').
        quality: JPEG compression quality (1-95). Higher is better quality, larger size.

    Returns:
        A base64 encoded string.
    """
    buffered = io.BytesIO()
    # Save the image to the in-memory buffer, specifying format and quality
    image_object.save(buffered, format=format, quality=quality)
    # Get the bytes from the buffer
    img_bytes = buffered.getvalue()
    # Base64 encode the bytes and decode to a UTF-8 string
    img_b64_bytes = base64.b64encode(img_bytes)
    img_b64_str = img_b64_bytes.decode('utf-8')
    return img_b64_str


## NEED TO MAKE THIS ASYNC an async client
class LLMProvider:
    """
    Convert a PDF dictionary page to PNG and send it to GPT-5-mini
    as a base64 image attachment.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        api_key: Optional[str] = None
    ):
        self.client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        self.model = model
        self.max_retries = max_retries

        self.system_prompt = (PROMPT_DIR / "system.txt").read_text("utf-8")
        self.user_prompt = (PROMPT_DIR / "user.txt").read_text("utf-8")

    # -------------------------------------------------------------
    # PDF → JPEG (optimized for GPT-5-mini)
    # -------------------------------------------------------------
    def _pdf_to_jpeg_base64(self, pdf_path: Path) -> str:
        pages = convert_from_path(str(pdf_path), dpi=300)

        if not pages:
            raise Exception(f"No pages in PDF: {pdf_path}")

        img = pages[0]

        # Convert to RGB JPEG
        jpeg_path = pdf_path.with_suffix(".jpg")
        img = img.convert("RGB")
        img.save(jpeg_path, "JPEG", quality=70)   # optimized size

        # Base64 encode
        b64 = pil_to_base64(img)
        return b64

    # -------------------------------------------------------------
    # Digitize page
    # -------------------------------------------------------------
    async def digitize_pdf(self, pdf_path: Path):
        jpeg_b64 = self._pdf_to_jpeg_base64(pdf_path)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{jpeg_b64}",
                        }
                    }
                ]
            }
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages  # DO NOT send temperature for gpt-5-mini
        )
        return json.loads(resp.choices[0].message.content)
