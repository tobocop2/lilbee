/** A failure the launcher can name: the CLI's wording in `message`, a stable `code` for embedders. */
export class LauncherError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = "LauncherError";
    this.code = code;
    Object.assign(this, details);
  }
}
