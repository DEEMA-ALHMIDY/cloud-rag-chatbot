resource "azurerm_postgresql_flexible_server" "db" {
  name                          = "deema-chatbot-db"
  resource_group_name           = data.azurerm_resource_group.rg.name
  location                      = data.azurerm_resource_group.rg.location
  version                       = "17"
  public_network_access_enabled = true

  administrator_login    = "weclouddata"
  administrator_password = var.db_admin_password

  storage_mb   = 32768
  storage_tier = "P4"

  sku_name = "B_Standard_B1ms"
}

resource "azurerm_postgresql_flexible_server_database" "appdb" {
  name      = "appdb"
  server_id = azurerm_postgresql_flexible_server.db.id
  collation = "en_US.utf8"
  charset   = "utf8"
}


output "db_endpoint" {
  value = azurerm_postgresql_flexible_server.db.fqdn
}