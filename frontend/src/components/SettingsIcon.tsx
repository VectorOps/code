const paths = {
  temporary: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7v5l3 2" />
    </>
  ),
  workspace: (
    <path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
  ),
  apply: <path d="m5 12 4 4L19 6" />,
  reset: (
    <>
      <path d="M3 5v5h5" />
      <path d="M3.7 10a8 8 0 1 1 1.8 8" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 4v5h-5M4 20v-5h5" />
      <path d="M4.7 9a8 8 0 0 1 13.2-3L20 9M4 15l2.1 3A8 8 0 0 0 19.3 15" />
    </>
  ),
};

export function SettingsIcon({ name }: { name: keyof typeof paths }) {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
