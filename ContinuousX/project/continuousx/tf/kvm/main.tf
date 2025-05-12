resource "openstack_networking_network_v2" "private_net" {
  name                  = "private-net-news-recommendation-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-news-recommendation-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

# Update ports with new naming convention
resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-news-recommendation-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet2_ports" {
  for_each   = var.nodes
    name       = "sharednet2-${each.key}-news-recommendation-${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet2.id
    security_group_ids = [
      data.openstack_networking_secgroup_v2.allow_ssh.id,
      data.openstack_networking_secgroup_v2.allow_9001.id,
      data.openstack_networking_secgroup_v2.allow_8000.id,
      data.openstack_networking_secgroup_v2.allow_8080.id,
      data.openstack_networking_secgroup_v2.allow_8081.id,
      data.openstack_networking_secgroup_v2.allow_http_80.id,
      data.openstack_networking_secgroup_v2.allow_9090.id,
      # Add news recommendation specific ports
      data.openstack_networking_secgroup_v2.allow_news_api.id,
      data.openstack_networking_secgroup_v2.allow_recommendation_engine.id
    ]
}

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name        = "${each.key}-news-recommendation-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  # Consider using a larger flavor for recommendation engine nodes
  flavor_name = each.key == "node1" ? "m1.large" : "m1.medium"
  key_pair    = var.key

  network {
    port = openstack_networking_port_v2.sharednet2_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-news-recommendation-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "News Recommendation IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet2_ports["node1"].id
}