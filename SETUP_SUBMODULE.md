# Setting up GPU Scheduler as a Git Submodule

The `gpu_scheduler` package has been initialized as its own git repository. To use it as a submodule in the parent project, follow these steps:

## Step 1: Push to Private Remote Repository

First, create a private repository (e.g., on GitHub, GitLab, or your private git server) and push the gpu_scheduler code:

```bash
cd gpu_scheduler

# Add your private remote (replace with your actual repository URL)
git remote add origin <your-private-repo-url>

# Push to remote
git push -u origin master
```

## Step 2: Remove Local Directory and Add as Submodule

After pushing to the remote, remove the local directory and add it as a submodule:

```bash
# From the parent repository root
cd /home/konpat/pytorch-CycleGAN-and-pix2pix

# Remove the local directory (it's already in .gitignore)
rm -rf gpu_scheduler

# Add as submodule (replace with your actual repository URL)
git submodule add <your-private-repo-url> gpu_scheduler

# Initialize and update
git submodule update --init --recursive
```

## Step 3: Verify

Verify the submodule is set up correctly:

```bash
# Check submodule status
git submodule status

# Verify imports work
python -c "from gpu_scheduler import GPUScheduler; print('✅ Import successful!')"
```

## Updating the Submodule

When you make changes to the gpu_scheduler repository:

```bash
# Update to latest version
git submodule update --remote gpu_scheduler

# Or update to a specific commit/tag
cd gpu_scheduler
git checkout <commit-hash-or-tag>
cd ..
git add gpu_scheduler
git commit -m "Update gpu_scheduler submodule"
```

## Cloning the Parent Repository with Submodules

When someone clones the parent repository, they need to initialize submodules:

```bash
# Clone with submodules
git clone --recursive <parent-repo-url>

# Or if already cloned
git submodule update --init --recursive
```

