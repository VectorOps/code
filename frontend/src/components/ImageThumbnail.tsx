import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useFocusBlocker } from "../focus/FocusProvider";
import "./images.css";

function ImagePreview({
  src,
  name,
  close,
}: {
  src: string;
  name: string;
  close: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useFocusBlocker(true, true);
  useLayoutEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  return createPortal(
    <dialog
      ref={dialog}
      className="image-preview"
      aria-label={`Image preview: ${name}`}
      onCancel={(event) => {
        event.preventDefault();
        close();
      }}
      onClose={close}
      onClick={(event) => {
        if (event.target === event.currentTarget) close();
      }}
    >
      <div className="image-preview-header">
        <span>{name}</span>
        <button
          type="button"
          className="quiet-button"
          aria-label="Close image preview"
          onClick={close}
        >
          Close
        </button>
      </div>
      <img src={src} alt={name} />
    </dialog>,
    document.body,
  );
}

export function ImageThumbnail({ src, name }: { src: string; name: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className="quiet-button image-thumbnail"
        aria-label={`Preview ${name}`}
        title={`Preview ${name}`}
        onClick={() => setOpen(true)}
      >
        <img src={src} alt={name} />
      </button>
      {open && (
        <ImagePreview src={src} name={name} close={() => setOpen(false)} />
      )}
    </>
  );
}
