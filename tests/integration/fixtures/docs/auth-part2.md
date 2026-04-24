# Authentication: JWT Tokens

JWT tokens are signed with RS256 algorithm using asymmetric keys.
Access tokens expire after 15 minutes. Refresh tokens are stored
securely and rotated on each use for replay attack prevention.
