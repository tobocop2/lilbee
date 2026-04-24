# Authentication: Session Management

Sessions stored in Redis with 24h TTL. Session IDs are generated
using cryptographically secure random bytes. Inactive sessions
are garbage collected every 6 hours by the cleanup worker.
