# Deployment Guide

Use kubectl apply to deploy containers to the Kubernetes cluster.
The CI/CD pipeline builds Docker images tagged with the git SHA.
Rolling updates ensure zero downtime during releases. Configure
resource limits and health checks in the deployment manifest.
