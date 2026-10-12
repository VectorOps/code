import type { StepProps } from "./registry";
import { StepIndicators } from "./StepIndicators";
import { memo } from "react";
import { imageSource, messageImages } from "../../models/chat-input";
import { ImageThumbnail } from "../ImageThumbnail";

export const UserStep = memo(function UserStep({ step, actions }: StepProps) {
  return (
    <section className="feed-entry user-step">
      <div className="assistant-controls">
        <StepIndicators inspect={() => actions.inspect(step.id)} />
      </div>
      <div className="message-text">
        {step.message || "Message with attachments"}
      </div>
      <div className="message-images">
        {messageImages(step.output).map((image, index) => (
          <ImageThumbnail
            key={index}
            src={imageSource(image)}
            name={`Attached image ${index + 1}`}
          />
        ))}
      </div>
    </section>
  );
});
