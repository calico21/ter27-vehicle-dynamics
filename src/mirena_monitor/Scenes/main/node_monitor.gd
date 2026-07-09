extends RefCounted
class_name NodeMonitor

var _node: RosNode
var _heartbeart_sub: RosSubscriber

var _col_rect: ColorRect
var _target: String


var tiempo_anterior: float = 0.0
const limite_tiempo: float = 2.0 # Seconds


func _init(col_rect: ColorRect, target_node_name: String) -> void:
	_col_rect = col_rect
	_target = target_node_name
	
	_change_color(0)

	_node = RosNode.new()
	_node.init("mirena_monitor", "")
	
	var sensor_qos = RosQoS.new()
	sensor_qos.reliability = RosQoS.BEST_EFFORT
	
	_heartbeart_sub = _node.create_subscription("health",RosMirenaCommonHeartbeat, _callback,sensor_qos)
	
	pass	

func update() -> void:
	var tiempo_ahora = Time.get_unix_time_from_system()
	if (tiempo_ahora-tiempo_anterior) > limite_tiempo:
		_change_color(0)

func _callback(msg: RosMirenaCommonHeartbeat):
	print("Listo")
	
	if msg.node_name == _target :
		tiempo_anterior = Time.get_unix_time_from_system()
		_change_color(int(msg.estado.status))

func _change_color(status: int):
	match (status):
		3:
			_col_rect.color = Color.YELLOW
		2:
			_col_rect.color = Color.ORANGE
		1:
			_col_rect.color = Color.GREEN_YELLOW
		0:
			_col_rect.color = Color.ORANGE_RED
