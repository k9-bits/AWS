packer {
  required_plugins {
    amazon = {
      version = ">= 1.0.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "aws_region" {
  default = "ap-southeast-2"
}

variable "hf_token" {
  description = "Hugging Face access token"
  type        = string
  default = ""
}


source "amazon-ebs" "dlami-base" {
  region           = var.aws_region
  source_ami       = "ami-006e788b82e1845de"    # Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2)
  instance_type    = "g5.2xlarge"              # GPU-compatible instance type
  ssh_username     = "ec2-user"
  ami_name         = "dlami-custom-{{timestamp}}"
  ssh_interface    = "public_ip"
}

build {
  name    = "build-custom-dlami"
  sources = ["source.amazon-ebs.dlami-base"]

  provisioner "shell" {
    inline = [
      "echo '🔧 Starting customization...'",
      "sudo yum update -y",
      "pip install --upgrade pip",
      "pip install torch torchvision torchaudio",
      "echo 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))' > check_cuda.py",
	  "python check_cuda.py",
      "source activate pytorch",
      "conda config --show-sources",
      "conda config --remove channels https://aws-ml-conda-ec2.s3.us-west-2.amazonaws.com",
      "conda update -n base conda -y",
      "pip install transformers datasets accelerate huggingface_hub",
      "pip install bitsandbytes accelerate",
      "huggingface-cli login --token $HF_TOKEN",
      "echo \"HF_TOKEN=$HF_TOKEN\" | sudo tee -a /etc/environment",
      "echo 'export HF_TOKEN=$HF_TOKEN' | sudo tee -a /etc/profile.d/hf_token.sh",
      "sudo chmod +x /etc/profile.d/hf_token.sh",
      "sudo systemctl enable docker",
      "sudo systemctl start docker",
      "echo '✅ Done!'"
    ]
    environment_vars = ["HF_TOKEN=${var.hf_token}"]
  }
}
