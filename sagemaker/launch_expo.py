import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import boto3
import tyro
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.aws_batch.training_queue import TrainingQueue as Queue
from sagemaker.session import Session as SageMakerSession

NAME = "expo"

INSTANCE_MAPPER = {
    #"p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "p5en": "ml.p5en.48xlarge",
    "p6": "ml.p6-b200.48xlarge",
}

QUEUE_MAPPER = {
    "us-west-2": {
        "ml.p5.48xlarge": "fss-ml-p5-48xlarge-us-west-2",
        "ml.p5en.48xlarge": "fss-vla-p5en-48xlarge-us-west-2",
       # "ml.p4de.24xlarge": "fss-ml-p4de-24xlarge-us-west-2",
        #"ml.p4d.24xlarge": "fss-ml-p4d-24xlarge-us-west-2",
        "ml.p6-b200.48xlarge": "fss-ml-p6-b200-48xlarge-us-west-2",
    },
}

def run_command(command: str) -> None:
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)

def sanitize_name(name: str) -> str:
    name = name.replace("_", "-")
    clean = "".join(c if c.isalnum() or c == "-" else "" for c in name)
    clean = clean.strip("-")
    return clean or "job"

def get_job_name(base: str) -> str:
    now = datetime.now()
    # Format example: 2023-03-03-10-14-02-324
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}"
    # Ensure the job name follows SageMaker naming constraints: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}
    clean_base = sanitize_name(base)
    job_name = f"{clean_base}-{date_str}"
    job_name = job_name.lstrip("-")
    # Truncate if too long (SageMaker limit is 63 characters)
    if len(job_name) > 63:
        job_name = job_name[:63]
    # Remove trailing hyphens if any (truncation may have left some)
    job_name = job_name.rstrip("-")
    return job_name

def get_image(user: str, profile: "default", region: "us-west-2") -> str:
    os.environ["AWS_PROFILE"] = profile
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    print("Account: ", account)
    assert account.isdigit(), f"Invalid account value: {account}"

    docker_dir = Path(__file__).parent
    repo_root = docker_dir.parent  # assumes sagemaker/launch_expo.py inside repo
    dockerfile = docker_dir / "Dockerfile.expo"

    algorithm_name = f"{user}-{NAME}"
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin"
    )

    commands = [
        # login to SM base ECR if needed (optional; keep for TRI parity)
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build -f {dockerfile} -t {algorithm_name} {repo_root}",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} --no-cli-pager || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name} --no-cli-pager"
        ),
    ]

    run_command("\n".join([f"{c} || exit 1" for c in commands]))
    run_command(f"docker push {fullname}")
    print("Sleeping 5s to ensure push succeeded...")
    time.sleep(5)
    return fullname


@dataclass(frozen=True)
class Args:
    # Experiment selection
    script: str = "expo_transport.sh"                 # scripts/expo/<script>
    seed: str | None = None                          # default: timestamp
    expo_args: str = ""                              # appended to script call, e.g. "--use_success_buffer=True"

    # Identity / naming
    user: str = "paarth.shah"
    name_prefix: str | None = None

    # AWS/SageMaker
    local: bool = False
    region: str = "us-west-2"
    profile: str = "default"
    arn: str = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess"

    instance_count: int = 1
    instance_type: str = "p5"
    max_run_days: int = 5
    volume_size: int = 200

    queue_name: str = "ml"
    priority: int = 100

    # Optional: load secrets from file
    secrets_env: str = "secrets.env"                 # contains WANDB_API_KEY=..., etc.


def main():
    args = tyro.cli(Args)

    # Override profile from environment if set (like the working file does)
    if "AWS_PROFILE" in os.environ and args.profile == "default":
        object.__setattr__(args, "profile", os.environ["AWS_PROFILE"])

    assert args.instance_type in INSTANCE_MAPPER
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        object.__setattr__(args, "arn", os.environ["SAGEMAKER_ARN"])

    # Set AWS_PROFILE in environment before any boto3 calls (like the working file)
    # Override from environment if set, otherwise use args.profile
    if "AWS_PROFILE" in os.environ:
        object.__setattr__(args, "profile", os.environ["AWS_PROFILE"])
    else:
        os.environ["AWS_PROFILE"] = args.profile

    image_uri = get_image(args.user, profile=args.profile, region=args.region)
    # Ensure environment variables are set (get_image sets AWS_PROFILE, but ensure it persists)
    os.environ["AWS_PROFILE"] = args.profile
    os.environ["AWS_DEFAULT_REGION"] = args.region

    sagemaker_session = SageMakerSession(boto_session=boto3.Session(region_name=args.region))
    if args.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()
    else:
        sagemaker_session = sagemaker.Session(
            boto_session=boto3.session.Session(region_name=args.region)
        )

    role = args.arn

    base_job_name = sanitize_name(
        f"{(args.name_prefix + '-') if args.name_prefix else ''}{args.user.replace('.', '-')}-{NAME}"
    )
    job_name = get_job_name(base_job_name)

    seed = args.seed or datetime.now().strftime("%Y%m%d%H%M%S")

    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        # Your train.sh contract
        "EXPO_SCRIPT": args.script,
        "SEED": seed,
        "EXPO_ARGS": args.expo_args,
        "USER": args.user,  # so /data/user_data/$USER/... looks sane

        # Common runtime knobs
        "MUJOCO_GL": "egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "NCCL_DEBUG": "INFO",
        "FI_EFA_FORK_SAFE": "1",
    }

    # Optionally merge secrets.env (WANDB_API_KEY, WANDB_ENTITY, etc.)
    secrets_path = Path(args.secrets_env)
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            environment[k.strip()] = v.strip().strip("\"'")
    else:
        print(f"Note: secrets file not found: {secrets_path} (ok if using other auth methods)")

    estimator = Estimator(
        entry_point="sagemaker/train.sh",  # exists inside image under /opt/ml/code
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        role=role,
        image_uri=image_uri,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        input_mode="FastFile",
        max_run=args.max_run_days * 24 * 60 * 60,
        environment=environment,
        keep_alive_period_in_seconds=5 * 60,
        volume_size=args.volume_size,
        tags=[
            {"Key": "tri.project", "Value": "EXPO"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    if args.local:
        # Local mode: just fit directly
        estimator.fit(inputs=None, job_name=job_name)
        print(f"Started local job {job_name}")
        return

    queue = Queue(
        queue_name=QUEUE_MAPPER[args.region][INSTANCE_MAPPER[args.instance_type]].replace("ml", args.queue_name)
    )
    queue.map(
        estimator,
        inputs=[None],
        job_names=[job_name],
        priority=args.priority,
        share_identifier="default",
        timeout={"attemptDurationSeconds": args.max_run_days * 24 * 60 * 60},
    )
    print(f"Queued {job_name} with script={args.script} seed={seed} expo_args='{args.expo_args}'")

if __name__ == "__main__":
    main()
