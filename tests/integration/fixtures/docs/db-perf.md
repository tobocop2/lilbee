# Database Performance

Index your queries. Use connection pooling with PgBouncer to reduce
connection overhead. Monitor slow queries with pg_stat_statements.
Vacuum and analyze tables regularly. Partition large tables by date
for faster range scans.
