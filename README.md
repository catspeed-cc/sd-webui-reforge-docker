
# Stable Diffusion WebUI Forge/reForge

Stable Diffusion WebUI Forge/reForge is a platform on top of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (based on [Gradio](https://www.gradio.app/)) to make development easier, optimize resource management, speed up inference, and study experimental features.

The name "Forge" is inspired from "Minecraft Forge". This project is aimed at becoming SD WebUI's Forge.

# Forge2/reForge2

You can read more on https://github.com/Panchovix/stable-diffusion-webui-reForge/discussions/377#discussioncomment-14010687. You can tell me here if you want to keep these branches here or do something like "reForge2".

* newmain_newforge: Based on latest forge2 (gradio4, flux, etc) with some small changes that I plan to add very slowly. For now it has python 3.12 support, sage/flash attention support, all the samplers and schedulers from reForge (1), and recently, support for CFG++ samplers.
* newforge_dendev: Based on latest ersatzForge fork which is based on forge2 (gradio4, flux, chroma, cosmos, longclip, and a ton more) from @DenOfEquity (https://github.com/DenOfEquity/ersatzForge). Many thanks Den for letting me to work on base on your fork on reForge. I will try to add new features from old reforge as well, like all the samplers.

# Suggestion: For stability based on old forge, use forge classic

reForge(1) is not really stable for all tasks sadly.

So if you want to keep using old forge backend as it is, for sd1.x,2.x and SDXL, I suggest to use forge classic by @Haoming02 instead https://github.com/Haoming02/sd-webui-forge-classic, as at the moment that is the real succesor to old forge.

Other branches:
* main: Main branch with multiple changes and updates. But not stable as main-old branch.
* dev: Similar to main but with more unstable changes. I.e. using comfy/ldm_patched backend for sd1.x and sdxl instead of A1111.
* dev2: More unstable than dev, for now same as dev.
* experimental: same as dev2 but with gradio 4.
* main-old: Branch with old forge backend. Possibly the most stable and older one (2025-03)

# Installing Forge/reForge

### (Suggested) Clean install.

For this, you will need Python (Python 3.7 up to 3.12 works fine, 3.13 still has some issues)
If you know what you are doing, you can install Forge/reForge using same method as SD-WebUI. (Install Git, Python, Git Clone the reForge repo `https://github.com/Panchovix/stable-diffusion-webui-reForge.git` and then run webui-user.bat):

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
```
Then run webui-user.bat (Windows) or webui-user.sh (Linux, for this one make sure to uncomment the lines according of your folder, paths and setting you need).

When you want to update:
```bash
cd stable-diffusion-webui-reForge
git pull
```

### If using Windows 7 and/or CUDA 11.x

For this, way to install is a bit different, since it uses another req file. We will rename the original req file to a backup, and then copy the legacy one renmaed as the original, to keep updates working.
For Windows CMD, it would be:

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
ren requirements_versions.txt requirements_versions_backup.txt
copy requirements_versions_legacy.txt requirements_versions.txt
```

Windows PS1

```bash
git clone https://github.com/Panchovix/stable-diffusion-webui-reForge.git
cd stable-diffusion-webui-reForge
git checkout main
Rename-Item requirements_versions.txt requirements_versions_backup.txt
Copy-Item requirements_versions_legacy.txt requirements_versions.txt
```

Then run webui-user.bat (Windows).

### You have A1111 and you know Git
Tutorial from: https://github.com/continue-revolution/sd-webui-animatediff/blob/forge/master/docs/how-to-use.md#you-have-a1111-and-you-know-git
If you have already had OG A1111 and you are familiar with git, An option is go to `/path/to/stable-diffusion-webui` and
```bash
git remote add reForge https://github.com/Panchovix/stable-diffusion-webui-reForge
git branch Panchovix/main
git checkout Panchovix/main
git fetch reForge
git branch -u reForge/main
git stash
git pull
```
To go back to OG A1111, just do `git checkout master` or `git checkout main`.

If you got stuck in a merge to resolve conflicts, you can go back with `git merge --abort`

-------

Pre-done package is planned, but I'm not sure how to do it. Any PR or help with this is appreciated.

# Forge/reForge Backend

Forge/reForge backend removes all WebUI's codes related to resource management and reworked everything. All previous CMD flags like `medvram, lowvram, medvram-sdxl, precision full, no half, no half vae, attention_xxx, upcast unet`, ... are all **REMOVED**. Adding these flags will not cause error but they will not do anything now.

Without any cmd flag, Forge/reForge can run SDXL with 4GB vram and SD1.5 with 2GB vram.

**Some flags that you may still pay attention to:** 

1. `--always-offload-from-vram` (This flag will make things **slower** but less risky). This option will let Forge/reForge always unload models from VRAM. This can be useful if you use multiple software together and want Forge/reForge to use less VRAM and give some VRAM to other software, or when you are using some old extensions that will compete vram with Forge/reForge, or (very rarely) when you get OOM.

2. `--cuda-malloc` (This flag will make things **faster** but more risky). This will ask pytorch to use *cudaMallocAsync* for tensor malloc. On some profilers I can observe performance gain at millisecond level, but the real speed up on most my devices are often unnoticed (about or less than 0.1 second per image). This cannot be set as default because many users reported issues that the async malloc will crash the program. Users need to enable this cmd flag at their own risk.

3. `--cuda-stream` (This flag will make things **faster** but more risky). This will use pytorch CUDA streams (a special type of thread on GPU) to move models and compute tensors simultaneously. This can almost eliminate all model moving time, and speed up SDXL on 30XX/40XX devices with small VRAM (eg, RTX 4050 6GB, RTX 3060 Laptop 6GB, etc) by about 15\% to 25\%. However, this unfortunately cannot be set as default because I observe higher possibility of pure black images (Nan outputs) on 2060, and higher chance of OOM on 1080 and 2060. When the resolution is large, there is a chance that the computation time of one single attention layer is longer than the time for moving entire model to GPU. When that happens, the next attention layer will OOM since the GPU is filled with the entire model, and no remaining space is available for computing another attention layer. Most overhead detecting methods are not robust enough to be reliable on old devices (in my tests). Users need to enable this cmd flag at their own risk.

4. `--pin-shared-memory` (This flag will make things **faster** but more risky). Effective only when used together with `--cuda-stream`. This will offload modules to Shared GPU Memory instead of system RAM when offloading models. On some 30XX/40XX devices with small VRAM (eg, RTX 4050 6GB, RTX 3060 Laptop 6GB, etc), I can observe significant (at least 20\%) speed-up for SDXL. However, this unfortunately cannot be set as default because the OOM of Shared GPU Memory is a much more severe problem than common GPU memory OOM. Pytorch does not provide any robust method to unload or detect Shared GPU Memory. Once the Shared GPU Memory OOM, the entire program will crash (observed with SDXL on GTX 1060/1050/1066), and there is no dynamic method to prevent or recover from the crash. Users need to enable this cmd flag at their own risk.

Some extra flags that can help with performance or save VRAM, or more, depending of your needs. Most of them are found on ldm_patched/modules/args_parser.py and on the normal A1111 path (modules/cmd_args.py):

    --disable-xformers
        Disables xformers, to use other attentions like SDP.
    --use-sage-attention
        Uses SAGE attention implementation, from https://github.com/thu-ml/SageAttention. You need to install the library separately, as it needs triton.
    --attention-split
        Use the split cross attention optimization. Ignored when xformers is used.
    --attention-quad
        Use the sub-quadratic cross attention optimization . Ignored when xformers is used.
    --attention-pytorch
        Use the new pytorch 2.0 cross attention function.
    --disable-attention-upcast
        Disable all upcasting of attention. Should be unnecessary except for debugging.
    --force-channels-last
        Force channels last format when inferencing the models.
    --disable-cuda-malloc
        Disable cudaMallocAsync.
    --gpu-device-id
        Set the id of the cuda device this instance will use.
    --force-upcast-attention
        Force enable attention upcasting.

(VRAM related)

    --always-gpu
        Store and run everything (text encoders/CLIP models, etc... on the GPU).
    --always-high-vram
        By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.
    --always-normal-vram
        Used to force normal vram use if lowvram gets automatically enabled.
    --always-low-vram
        Split the unet in parts to use less vram.
    --always-no-vram
        When lowvram isn't enough.
    --always-cpu
        To use the CPU for everything (slow).

(float point type)

    --all-in-fp32
    --all-in-fp16
    --unet-in-bf16
    --unet-in-fp16
    --unet-in-fp8-e4m3fn
    --unet-in-fp8-e5m2
    --vae-in-fp16
    --vae-in-fp32
    --vae-in-bf16
    --clip-in-fp8-e4m3fn
    --clip-in-fp8-e5m2
    --clip-in-fp16
    --clip-in-fp32

(rare platforms)

    --directml
    --disable-ipex-hijack
    --pytorch-deterministic

# Lora ctl (Control)

I've added this repo adapted for reforge.

This wouldn't be possible to do without the original ones!

Huge credits to cheald for Lora ctl (Control). Link for the reforge extension is: https://github.com/Panchovix/sd_webui_loractl_reforge_y.git

Many thanks to @1rre for his work for preliminary working version for lora control!

You can see how to use them on their respective repos

https://github.com/cheald/sd-webui-loractl

## Moved built-it extensions to separate repos

Since the UI got really cluttered with built it extensions, I have removed some of them and made them separate repos. You can install them by the extension installer on the UI or doing `git clone repo.git` replacing `repo.git` with the following links, in the extensions folder.

* RAUNet-MSW-MSA (HiDiffusion): https://github.com/Panchovix/reforge_jankhidiffusion.git
* Skimmed CFG: https://github.com/Panchovix/reForge-SkimmedCFG.git
* Forge Style Align: https://github.com/Panchovix/sd_forge_stylealign.git
* reForge Sigmas Merge: https://github.com/Panchovix/reForge-Sigmas_merge.git
* Differential Diffusion: https://github.com/Panchovix/reForge-DifferentialDiffusion.git
* Auomatic CFG: https://github.com/Panchovix/reForge-AutomaticCFG.git
* reForge_Advanced_CLIP_Text_Encode (not working yet): https://github.com/Panchovix/reForge_Advanced_CLIP_Text_Encode.git
* Hunyuan-DiT-for-webUI-main: https://github.com/Panchovix/Hunyuan-DiT-for-webUI-main.git
* PixArt-Sigma-for-webUI-main: https://github.com/Panchovix/PixArt-Sigma-for-webUI-main.git
* StableCascade-for-webUI-main: https://github.com/Panchovix/StableCascade-for-webUI-main.git
* StableDiffusion3-for-webUI-main: https://github.com/Panchovix/StableDiffusion3-for-webUI-main.git

# Last "Old" Forge commit (https://github.com/lllyasviel/stable-diffusion-webui-forge/commit/bfee03d8d9415a925616f40ede030fe7a51cbcfd) before forge2.

# Support

Some people have been asking how to donate or support the project, and I'm really grateful for that! I did this buymeacoffe link from some suggestions!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/Panchovix)

# Docker Installation

### IMPORTANT FOR v1.1.0: Cuda version has been upgraded for container from 12.1 -> 12.8

NOTE: Please upgrade your driver first to 535.104.05 or higher

Cuda version has been upgraded for container from 12.1 -> 12.8 please install cuda 12.8 ON THE HOST machine (instructions below)

Both of these can be modified to suit other versions of debian/ubuntu

Debian 11
```
sudo apt-get remove --purge '^cuda.*' '^nvidia-cuda.*' && \
sudo apt-get autoremove -y && \
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-debian11.pin && \
sudo mv cuda-debian11.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub && \
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" && \
sudo apt-get update && \
sudo apt-get install -y cuda-toolkit-12-8
```

Ubuntu 22.04 change cuda 12.1 -> 12.8
```
sudo apt-get remove --purge '^cuda.*' '^nvidia-cuda.*' && \
sudo apt-get autoremove -y && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
sudo apt-get update && \
sudo apt-get install -y cuda-toolkit-12-8
```

### Docker Installation (cont'd)

Install Docker:
- `sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`
- Test the installation worked with `docker compose version` you should get something like `Docker Compose version v2.24.5`
- If trouble submit a [catspeed-cc issue ticket](https://github.com/catspeed-cc/sd-webui-forge-docker/issues)

Models can be put in `sd-webui-forge-docker/models/` directory, organized by type - they will be mounted to the container
- If you copy models in while container running and after first start, after model copy is complete you can `docker-stop-containers.sh` and `docker-start-containers.sh` and they will be loaded quickly

Outputs are stored in `sd-webui-forge-docker/outputs/` directory

Due to the nature of Docker, an image running at shutdown _should_ start up again on boot. If this does not happen, submit a [catspeed-cc issue ticket](https://github.com/catspeed-cc/sd-webui-forge-docker/issues)

These are the current tags:
```
catspeedcc/sd-webui-forge-docker:latest - currently points to v1.1.0
catspeedcc/sd-webui-forge-docker:v1.1.0 - Important upgrades: smaller docker image, cuda 12.1->12.8 (see README.md)
catspeedcc/sd-webui-forge-docker:v1.0.0 - latest stable version (first release)

catspeedcc/sd-webui-forge-docker:development - (not supported, parity w/ development branch, if you use it you're on your own.)
catspeedcc/sd-webui-forge-docker:bleeding - (not supported, ephemeral, if you use it you're on your own.)
```

There are a few main config files:
```
docker-compose.yaml # CPU-only          # Should not need configuration

docker-compose.single-gpu.nvidia.yaml   # Single GPU only (needs config)
docker-compose.multi-gpu.nvidia.yaml    # ONE OF MULTIPLE GPU only (needs config)

docker-compose.combined.nvidia.yaml     # ONLY so you can copy the service into
                                        # a different docker-compose.yml file ;)
```

As far as I know there is no way to combine multiple GPU's on this one same task (image generation) but you can dedicate one of many GPU's to image generation and then use the other GPU's for other tasks (chat, development, etc)

- Clone the catspeed-cc repository `git clone https://github.com/catspeed-cc/sd-webui-forge-docker.git`
- DO NOT run the `webui-docker.sh` script ever, it is meant ONLY to be ran inside the docker containers at runtime. (automatic, ignore the file)
- Read the rest of this section, then jump to either [CPU Only](https://github.com/catspeed-cc/sd-webui-forge-docker/README.md#cpu-only-untested), [Single GPU Only](https://github.com/catspeed-cc/sd-webui-forge-docker/README.md#single-gpu-only-untested-should-work), or [Single of Multiple GPU Only](https://github.com/catspeed-cc/sd-webui-forge-docker/README.md#single-of-multiple-gpu-only-tested)

_**Important:**_ All Docker support for now goes to [catspeed-cc issue tickets](https://github.com/catspeed-cc/sd-webui-forge-docker/issues) until and _only if_ this ever gets merged upstream.

### CPU Only (untested)

- `./docker-init-cpu-only.sh` "installs" and starts the docker container
- After install, even while running you can copy models to models/ and then after run stop/start for quick reload
- `./docker-stop-containers.sh` "stops" container(s)
- `./docker-reinstall-container-deps.sh` - reinstalls containers dependencies (requires stop/start, you should prefer to destroy/init)
- `./docker-start-containers.sh` "starts" container(s)
- `./docker-destroy-cpu-only.sh` "uninstalls" and stops the docker container
- You can uninstall/reinstall to debug / start with fresh image (image is already stored locally)

### Single GPU Only (untested, should work)

- Edit & configure `docker-compose.single-gpu.yaml`
- `./docker-init-single-gpu-only.sh` "installs" and starts the docker container
- After install, even while running you can copy models to models/ and then after run stop/start for quick reload
- `./docker-stop-containers.sh` "stops" container(s)
- `./docker-reinstall-container-deps.sh` - reinstalls containers dependencies (requires stop/start, you should prefer to destroy/init)
- `./docker-start-containers.sh` "starts" container(s)
- `./docker-destroy-single-gpu-only.sh` "uninstalls" and stops the docker container
- You can uninstall/reinstall to debug / start with fresh image (image is already stored locally)

### Single of Multiple GPU Only (tested)

- Edit & configure `docker-compose.multi-gpu.yaml`
- `./docker-init-multi-gpu-only.sh` "installs" and starts the docker container
- After install, even while running you can copy models to models/ and then after run stop/start for quick reload
- `./docker-stop-containers.sh` "stops" container(s)
- `./docker-reinstall-container-deps.sh` - reinstalls containers dependencies (requires stop/start, you should prefer to destroy/init)
- `./docker-start-containers.sh` "starts" container(s)
- `./docker-destroy-multi-gpu-only.sh` "uninstalls" and stops the docker container
- You can uninstall/reinstall to debug / start with fresh image (image is already stored locally)

### Customize a docker-compose.yaml from another project
Let's say you have another project - let's pick localAGI as an example. You can customize the `docker-compose.yaml` for localAGI and add in this docker service. This way when you start localAGI it will also start your image generation service.

- Open the localAGI (or other project) directory
- Download the sauces archive for your version from https://github.com/catspeed-cc/sd-webui-forge-docker/tree/master/sauces
- Extract the sauces into your localAGI (or other) project directory `tar zxvf v1.0.0-sauce.tar.gz`
- Edit & configure `docker-compose.combined.nvidia.yaml`
- Copy the lines for the service from `docker-compose.combined.nvidia.yaml`
- Paste the lines underneath one of the other services inside the localAGI (or other project) docker-compose.yaml
- All sauce helper scripts and docker-compose.yaml files should now be in your project :)
- DO NOT use the init/destroy scripts, use your `docker compose up` and `docker-compose down` commands as directed by the project `README.md`
- Docker helper start/stop scripts will speed up startup when simply stopping or starting the container quickly (ex. to load new models)
- IF you need to destroy the container and recreate it for debugging/troubleshooting, then use the respective destroy script followed by `docker compose down` in the localAGI (or other project)
- Sauce scripts ONLY will init/destroy/start/stop sd-forge containers - _assuming you did not rename the containers!_
- IF you chose to rename the container, just make sure "sd-forge" exists in the name, and the sauce scripts should still work :)

## Sauces Archives & Start-Stop Docker Helper Scripts:
The models have been removed from the sauces directory as there are no models in the upstream/main repository contrary to what I initially thought. This is good, it allows me to remove the models, slim the archive, and then host it on GitHub because the sauce archives are basically just text files zipped up.

- Each version (major or minor) will have a corresponding sauce archive.
- You only need this sauce archive IF you are planning to use the `docker-compose.combined.nvidia.yaml` to customize a different docker-compose.yaml and add sd-forge as a service.
- You _could_ use sauces to run a cut down installation that is standalone - no integration with a different docker-compose.yaml - extract, edit yaml, run the respective init script
- Due to the sauces being hosted on GitHub, MD5SUM's are not required (we are staying on the secured, confirmed, GitHub)
- MD5SUM's will be posted inside an .MD5 file as the helper script can do it automatically
- Checking MD5SUM is not required unless you are extremely paranoid

## Future Plans:

- None as of yet
- please submit suggestions to https://github.com/catspeed-cc/sd-webui-forge-docker/issues

## Docker Support Warning:
ONLY post docker related issue tickets on this repository's issue ticket system. Any issues with the main project are still to be opened in the main repository.

- ALL docker related issues shall be posted here: https://github.com/catspeed-cc/sd-webui-forge-docker/issues
- ALL `stable-diffusion-webui-forge` related issues shall be posted here: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues

## Startup Time Warning:

FIRST RUN (when you run the init/install script) is going to take a while - grab a coffee. This was the tradeoff for the smaller docker image. Afterwards, you can mostly use `docker-stop-containers.sh` and `docker-start-containers.sh` to restart containers quickly, ex. if you installed new models.

## Large Image Warning:

The image may not be _as large_ as it was but I will check if this section can be removed ...

The developers know their own project better than I - and I am a noob. They can integrate it into docker better, and try to cut waste out of the image, but of course all dependencies need to be baked into the image. Otherwise the images will not work, or it would have to fetch them inside the container _every time_ you wanted to `docker compose down`. It is not the kind of image I would suggest converting to alpine to slim it down, it would be _a lot_ of work and _headache_. I am happy to help with anything, but mostly can sit and make my own mess in _my repository_ :)

Do not worry, I have _not_ loaded it with 1000's of models :P

## Docker Image Build Warning: (unsupported)

These are mostly for my reference. If you wish to build the image they are here for you also. Just keep in mind this is unsupported and you are on your own.

- `docker build -t myorganization/myrepository:mytag .` general build (will be cached)

**_OR_**

- `docker build --progress=plain --build-arg DUMMY=$(date +%s) -t myorganization/myrepository:mytag .` debug build - so you can debug the Dockerfile without caching certain elements

That's it! As previously mentioned, there is no support for this from this point onwards. 

These were documented for @mooleshacat (A.K.A. _future noob self_)
