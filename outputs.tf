output "vm_public_ip" {
  description = "Public IP address of the chatbot VM"
  value       = azurerm_public_ip.pip.ip_address
}

output "vm_name" {
  description = "Name of the chatbot virtual machine"
  value       = azurerm_linux_virtual_machine.vm.name
}