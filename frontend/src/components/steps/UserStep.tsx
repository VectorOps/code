import type { StepProps } from "./registry";
import { StepCard } from "./StepCard";
import { memo } from "react";
import { imageSource, messageImages } from "../../models/chat-input";
import { ImageThumbnail } from "../ImageThumbnail";

export const UserStep = memo(function UserStep({ step, actions }: StepProps) {
  return (
    <StepCard className="user-step" inspect={() => actions.inspect(step.id)}>
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
    </StepCard>
  );
});
