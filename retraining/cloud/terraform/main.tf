terraform {
  required_version = ">= 1.6.0"

  required_providers {
    yandex = {
      source  = "registry.terraform.io/yandex-cloud/yandex"
      version = "0.215.0"
    }
  }
}

provider "yandex" {
  folder_id = var.folder_id
  zone      = var.zone
}

data "yandex_compute_image" "ubuntu" {
  family = "ubuntu-2404-lts"
}

resource "yandex_compute_disk" "work" {
  name = "${var.instance_name}-data"
  zone = var.zone
  type = "network-hdd"
  size = var.work_disk_gib

  labels = {
    task = "ml-training"
  }
}

resource "yandex_compute_instance" "training" {
  name               = var.instance_name
  folder_id          = var.folder_id
  zone               = var.zone
  platform_id        = "standard-v3"
  service_account_id = var.service_account_id

  resources {
    cores         = var.cores
    memory        = var.memory_gib
    core_fraction = 100
  }

  boot_disk {
    auto_delete = true

    initialize_params {
      image_id = data.yandex_compute_image.ubuntu.id
      size     = 30
      type     = "network-ssd"
    }
  }

  secondary_disk {
    disk_id     = yandex_compute_disk.work.id
    device_name = "ml-data"
    auto_delete = false
  }

  network_interface {
    subnet_id          = var.subnet_id
    nat                = true
    security_group_ids = [var.security_group_id]
  }

  metadata = {
    user-data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
      bootstrap_b64 = filebase64("${path.module}/../bootstrap.py")
      config_b64 = base64encode(jsonencode({
        bucket = var.bucket_name
        run_id = var.run_id
      }))
    })
  }

  labels = {
    task = "ml-training"
    run  = var.run_id
  }
}

output "instance_id" {
  value = yandex_compute_instance.training.id
}

output "instance_name" {
  value = yandex_compute_instance.training.name
}
