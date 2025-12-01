# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Papilon, please report it by sending an email to **security@vector1.ai**.

**Please do NOT:**
- Open a public GitHub issue for security vulnerabilities
- Disclose the vulnerability publicly before it has been addressed

**Please DO include:**
- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (optional)

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days for critical issues

## Security Best Practices

When using Papilon:

1. **API Keys**: Never commit API keys to version control
   ```bash
   # Use environment variables
   export ANTHROPIC_API_KEY="sk-..."
   ```

2. **Data Handling**: Ensure PII is properly anonymized before processing

3. **Access Control**: Implement appropriate access controls for production deployments

4. **Updates**: Keep Papilon and its dependencies updated

5. **Audit Logging**: Enable audit logging for compliance requirements

## Responsible Disclosure

We follow a responsible disclosure policy. After a fix has been deployed, we will:
- Credit the reporter (unless they wish to remain anonymous)
- Publish a security advisory
- Update the changelog

Thank you for helping keep Papilon secure!
