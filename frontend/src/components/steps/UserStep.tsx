import type { StepProps } from "./registry";
import { StepCard } from "./StepCard";
import { memo } from "react";
import { imageSource, messageImages } from "../../models/chat-input";
import { ImageThumbnail } from "../ImageThumbnail";
import { useStepBranchControls } from "../StepBranchControls";

export const UserStep = memo(function UserStep({ step, actions }: StepProps) {
  const branch = useStepBranchControls(step, actions);
  return (
    <StepCard className="user-step" inspect={() => actions.inspect(step.id)}>
      {branch.icons && <div className="assistant-controls">{branch.icons}</div>}
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
      {branch.editor}
    </StepCard>
  );
});
