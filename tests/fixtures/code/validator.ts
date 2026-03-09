/**
 * Validation and sanitisation utilities for user input.
 */

const EMAIL_REGEX = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

/** Validate whether the input string is a well-formed email address. */
export function validateEmail(input: string): boolean {
    if (!input || input.length > 254) {
        return false;
    }
    return EMAIL_REGEX.test(input);
}

const HTML_TAG_REGEX = /<\/?[^>]+(>|$)/g;
const ENTITY_MAP: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
};

/** Strip HTML tags and escape dangerous characters from raw input. */
export function sanitizeHtml(raw: string): string {
    const stripped = raw.replace(HTML_TAG_REGEX, "");
    return stripped.replace(/[&<>"']/g, (ch) => ENTITY_MAP[ch] ?? ch);
}
