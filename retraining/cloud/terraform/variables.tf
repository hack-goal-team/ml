variable "folder_id" {
  type = string
}

variable "zone" {
  type    = string
  default = "ru-central1-a"
}

variable "subnet_id" {
  type = string
}

variable "security_group_id" {
  type = string
}

variable "service_account_id" {
  type = string
}

variable "bucket_name" {
  type = string
}

variable "run_id" {
  type = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{2,50}$", var.run_id))
    error_message = "run_id must be 3-51 lowercase letters, digits or hyphens."
  }
}

variable "instance_name" {
  type = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,62}$", var.instance_name))
    error_message = "instance_name must be a valid Yandex Compute Cloud VM name."
  }
}

variable "cores" {
  type    = number
  default = 16
}

variable "memory_gib" {
  type    = number
  default = 128
}

variable "work_disk_gib" {
  type    = number
  default = 450
}
