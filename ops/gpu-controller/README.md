# Swade GPU Controller

This controller lives on the permanent Hostinger VPS and treats Hostinger GPU instances as disposable compute workers.

## Why

Hostinger GPU instances currently cannot be safely paused while preserving the instance and stopping hourly billing. Therefore persistent state must live outside the GPU.

## Permanent state

Keep these outside the disposable GPU:

- Voice profiles and identity mappings
- Voicebox database
- Generated audio masters
- API credentials and app-specific access rules
- Job queue and project metadata
- Deployment branch and configuration
- Backblaze B2 backup copies

## Worker lifecycle

1. Deploy a GPU manually in Hostinger.
2. Register its current IP on the permanent VPS:

   `gpuctl register <GPU_IP>`

3. Provision it:

   `gpuctl provision`

4. Verify it:

   `gpuctl health`

5. Run voice generation jobs.
6. Before destroying the GPU:

   `gpuctl drain`

7. Confirm:

   `gpuctl safe-to-destroy`

   Continue only when the result is `SAFE_TO_DESTROY=YES`.

8. Destroy the GPU manually in Hostinger.
9. Clear the old worker registration:

   `gpuctl clear`

## Security

- Use a purpose-scoped SSH key on the GPU instance.
- Do not paste private SSH keys, root passwords, B2 secrets, or API tokens into chat.
- Keep B2 secrets in `/root/.config/swade-voice-b2.env` with mode 600.
- The GPU controller never destroys the GPU itself. Destruction remains an explicit elevated-approval action.

## Future automation

If Hostinger exposes a supported GPU create/destroy API or MCP action, it can be added behind explicit approval. Until then, instance creation/destruction stays manual while provisioning, health checks, backup, and drain are automated.
