import { object, text, type Json } from "./json";

export const imageTypes = [
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
] as const;
export const imageAccept = imageTypes.join(",");
export const maxImageBytes = 786432;
export const maxContentBlocks = 4;

export type ImageInput = {
  type: "image";
  mime_type: (typeof imageTypes)[number];
  data: string;
};

export type ImageAttachment = { id: string; name: string } & (
  | { state: "preparing" }
  | { state: "ready"; image: ImageInput }
  | { state: "failed"; error: string }
);

function isImageType(value: string): value is ImageInput["mime_type"] {
  return imageTypes.some((type) => type === value);
}

export function readImage(
  file: File,
  signal: AbortSignal,
): Promise<ImageInput> {
  if (!isImageType(file.type))
    return Promise.reject(new Error("Choose a PNG, JPEG, GIF, or WebP image."));
  const mime = file.type;
  if (file.size === 0 || file.size > maxImageBytes)
    return Promise.reject(
      new Error("Images must be nonempty and at most 768 KiB each."),
    );
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    const abort = () => {
      reader.abort();
      reject(new Error("Image preparation cancelled."));
    };
    reader.onloadend = () => signal.removeEventListener("abort", abort);
    reader.onerror = () =>
      reject(new Error("Could not read this image. Remove it and try again."));
    reader.onabort = () => reject(new Error("Image preparation cancelled."));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("Could not read this image."));
        return;
      }
      const data = reader.result.slice(reader.result.indexOf(",") + 1);
      resolve({ type: "image", mime_type: mime, data });
    };
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
    else reader.readAsDataURL(file);
  });
}

export function messageError(
  value: string,
  images: readonly ImageAttachment[],
  requestType?: string,
): string {
  if (images.some((image) => image.state === "preparing"))
    return "Preparing images…";
  if (images.some((image) => image.state === "failed"))
    return "Remove failed images before sending.";
  if (
    images.length &&
    requestType !== undefined &&
    requestType !== "chat.message.v1"
  )
    return "This workflow accepts text only. Remove the images to send.";
  return contentError(value, images.length);
}

function contentError(value: string, imageCount: number): string {
  if (Array.from(value).length > 65536)
    return "Messages can contain at most 65,536 text characters.";
  if (imageCount + (value.trim() ? 1 : 0) > maxContentBlocks)
    return "A message can contain at most four images, or three images with text.";
  return "";
}

export function chatMessage(
  value: string,
  images: readonly ImageInput[] = [],
): Json {
  const error = contentError(value, images.length);
  if (error) throw new Error(error);
  const content: Json[] = value.trim() ? [{ type: "text", text: value }] : [];
  content.push(...images);
  if (!content.length)
    throw new Error("Add a message or an image before sending.");
  return { content };
}

export function imageSource(image: ImageInput): string {
  return `data:${image.mime_type};base64,${image.data}`;
}

export function messageImages(output: Json | undefined): ImageInput[] {
  const content = object(output).content;
  if (!Array.isArray(content)) return [];
  return content.flatMap((block) => {
    const image = object(block);
    const mime = text(image.mime_type);
    const data = text(image.data);
    return image.type === "image" &&
      isImageType(mime) &&
      data.length >= 4 &&
      data.length <= 1048576 &&
      /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(
        data,
      )
      ? [{ type: "image" as const, mime_type: mime, data }]
      : [];
  });
}
