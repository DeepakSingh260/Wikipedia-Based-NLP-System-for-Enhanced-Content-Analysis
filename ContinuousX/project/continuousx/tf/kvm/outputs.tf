output "floating_ip_out" {
  description = "Floating IP assigned to node1"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

# Add new outputs
output "api_endpoint" {
  description = "News recommendation API endpoint"
  value       = "http://${openstack_networking_floatingip_v2.floating_ip.address}:5000"
}

output "monitoring_dashboard" {
  description = "Monitoring dashboard URL"
  value       = "http://${openstack_networking_floatingip_v2.floating_ip.address}:3000"
}

output "node_ips" {
  description = "IP addresses of all nodes"
  value       = {
    for k, v in openstack_compute_instance_v2.nodes : k => v.access_ip_v4
  }
}