import { ToolFormatterFactory, type ToolFormatter } from "./factory";
import * as views from "./views";

export const toolNames: Readonly<Record<string, string>> = {
  read_file: "Read File",
  list_files: "List Files",
  apply_patch: "Apply Patch",
  exec: "Run Command",
  poll_process: "Poll Process",
  write_stdin: "Send Process Input",
  kill_process: "Stop Process",
  list_processes: "List Processes",
  web_fetch: "Fetch Web Page",
  update_plan: "Update Plan",
  run_agent: "Run Agent",
  subworkflow: "Run Workflow",
};
const generic: ToolFormatter = {
  summary: views.GenericSummary,
  details: views.GenericDetails,
  width: 520,
};
const process: ToolFormatter = {
  summary: views.ProcessSummary,
  activity: { completed: "Ran", running: "Running" },
  details: views.ProcessDetails,
  width: 760,
};
const workflow: ToolFormatter = {
  summary: views.WorkflowSummary,
  details: views.GenericDetails,
  width: 520,
  visible: (tool) =>
    tool.failed || !["completed", "Authorized"].includes(tool.state),
};

export const defaultToolFormatters = new ToolFormatterFactory(
  toolNames,
  {
    read_file: {
      summary: views.FileSummary,
      activity: { completed: "Read", running: "Reading" },
      details: views.FileDetails,
      width: 760,
    },
    list_files: {
      summary: views.FileSummary,
      activity: { completed: "Listed", running: "Listing" },
      details: views.FilesDetails,
      width: 600,
    },
    apply_patch: {
      summary: views.PatchSummary,
      activity: { completed: "Requested patch", running: "Patching" },
      details: views.PatchDetails,
      width: 1120,
    },
    exec: process,
    poll_process: {
      ...process,
      activity: { completed: "Checked", running: "Checking" },
    },
    kill_process: {
      ...process,
      activity: { completed: "Stopped", running: "Stopping" },
    },
    write_stdin: {
      ...process,
      activity: { completed: "Sent input to", running: "Sending input to" },
      details: views.GenericDetails,
    },
    list_processes: { ...generic, details: views.ProcessesDetails },
    web_fetch: {
      summary: views.WebSummary,
      details: views.WebDetails,
      width: 760,
    },
    update_plan: {
      summary: views.PlanSummary,
      activity: { completed: "Updated plan", running: "Updating plan" },
      hideIndicators: true,
      details: views.PlanDetails,
      width: 520,
    },
    run_agent: workflow,
    subworkflow: workflow,
  },
  generic,
);
