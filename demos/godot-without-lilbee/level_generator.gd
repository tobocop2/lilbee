extends Node2D

@export var map_width: int = 40
@export var map_height: int = 30
@export var tile_size: Vector2i = Vector2i(16, 16)
@export var max_rooms: int = 8
@export var min_room_size: int = 4
@export var max_room_size: int = 10
@export var room_overlap_allowed: bool = false
@export var collectible_count: int = 10
@export var fill_probability: float = 0.4

enum TileType { FLOOR = 0, WALL = 1 }

var tilemap: TileMapLayer
var astar_grid: AStarGrid2D
var rooms: Array[Rect2i] = []
var collectible_positions: Array[Vector2i] = []
var rng: RandomNumberGenerator

func _ready() -> void:
	rng = RandomNumberGenerator.new()
	rng.randomize()
	_generate_level()

func _generate_level() -> void:
	_setup_tilemap()
	_setup_astar()
	_generate_rooms()
	_place_tiles()
	_scatter_collectibles()
	queue_redraw()

func _setup_tilemap() -> void:
	tilemap = TileMapLayer.new()
	tilemap.name = "TileMapLayer"
	add_child(tilemap)

func _setup_astar() -> void:
	astar_grid = AStarGrid2D.new()
	astar_grid.region = Rect2i(Vector2i.ZERO, Vector2i(map_width, map_height))
	astar_grid.cell_size = Vector2(tile_size)
	astar_grid.diagonal_mode = AStarGrid2D.DIAGONAL_MODE_NEVER
	astar_grid.heuristic = AStarGrid2D.HEURISTIC_MANHATTAN
	astar_grid.update()

func _generate_rooms() -> void:
	rooms.clear()
	var attempts: int = 0
	var max_attempts: int = 100

	while rooms.size() < max_rooms and attempts < max_attempts:
		attempts += 1
		var room_width: int = rng.randi_range(min_room_size, max_room_size)
		var room_height: int = rng.randi_range(min_room_size, max_room_size)
		var room_x: int = rng.randi_range(1, map_width - room_width - 1)
		var room_y: int = rng.randi_range(1, map_height - room_height - 1)
		
		var new_room := Rect2i(room_x, room_y, room_width, room_height)
		
		if _is_room_valid(new_room):
			_connected_new_room(new_room)
			rooms.append(new_room)

func _is_room_valid(room: Rect2i) -> bool:
	if not room_overlap_allowed:
		for existing_room in rooms:
			if room.intersects(existing_room.grow_individual(1)):
				return false
	return true

func _connected_new_room(new_room: Rect2i) -> void:
	if rooms.is_empty():
		return
	
	var prev_room: Rect2i = rooms.pick_random()
	var prev_center: Vector2i = prev_room.get_center()
	var new_center: Vector2i = new_room.get_center()
	
	if rng.bool():
		_connect_horizontal(prev_center, new_center)
		_connect_vertical(prev_center, new_center)
	else:
		_connect_vertical(prev_center, new_center)
		_connect_horizontal(prev_center, new_center)

func _connect_horizontal(from: Vector2i, to: Vector2i) -> void:
	var x_start: int = mini(from.x, to.x)
	var x_end: int = maxi(from.x, to.x)
	for x in range(x_start, x_end + 1):
		var floor_rect := Rect2i(x, from.y, 1, 1)
		_add_floor_tiles(floor_rect)

func _connect_vertical(from: Vector2i, to: Vector2i) -> void:
	var y_start: int = mini(from.y, to.y)
	var y_end: int = maxi(from.y, to.y)
	for y in range(y_start, y_end + 1):
		var floor_rect := Rect2i(to.x, y, 1, 1)
		_add_floor_tiles(floor_rect)

func _add_floor_tiles(rect: Rect2i) -> void:
	for y in range(rect.position.y, rect.end.y):
		for x in range(rect.position.x, rect.end.x):
			var pos := Vector2i(x, y)
			if _is_valid_tile_pos(pos) and not placed_tiles.has(pos):
				tilemap.set_cell(pos, 0, Vector2i(0, 0), 0)
				placed_tiles[pos] = TileType.FLOOR

var placed_tiles: Dictionary = {}

func _place_tiles() -> void:
	for y in range(map_height):
		for x in range(map_width):
			var pos := Vector2i(x, y)
			if not placed_tiles.has(pos):
				if _is_boundary(x, y) or rng.randf() < fill_probability:
					tilemap.set_cell(pos, 0, Vector2i(1, 0), 0)
					astar_grid.set_point_solid(pos, true)
					placed_tiles[pos] = TileType.WALL
				else:
					placed_tiles[pos] = TileType.FLOOR

func _is_boundary(x: int, y: int) -> bool:
	return x == 0 or x == map_width - 1 or y == 0 or y == map_height - 1

func _is_valid_tile_pos(pos: Vector2i) -> bool:
	return pos.x >= 0 and pos.x < map_width and pos.y >= 0 and pos.y < map_height

func _scatter_collectibles() -> void:
	collectible_positions.clear()
	var floor_positions: Array[Vector2i] = []
	
	for pos in placed_tiles:
		if placed_tiles[pos] == TileType.FLOOR:
			floor_positions.append(pos)
	
	floor_positions.shuffle()
	
	var start_pos: Vector2i = Vector2i(map_width / 2, map_height / 2)
	if placed_tiles.get(start_pos, TileType.WALL) == TileType.FLOOR:
		astar_grid.set_point_solid(start_pos, true)
		floor_positions.erase(start_pos)
	
	var reachable: Array[Vector2i] = _get_reachable_floor_tiles(start_pos)
	reachable.shuffle()
	
	for i in range(mini(collectible_count, reachable.size())):
		collectible_positions.append(reachable[i])

func _get_reachable_floor_tiles(start: Vector2i) -> Array[Vector2i]:
	var reachable: Array[Vector2i] = []
	var visited: Dictionary = {}
	var queue: Array[Vector2i] = [start]
	
	while not queue.is_empty():
		var current: Vector2i = queue.pop_front()
		if visited.has(current):
			continue
		visited[current] = true
		
		if not astar_grid.is_point_solid(current):
			reachable.append(current)
		
		var neighbors: Array[Vector2i] = [
			current + Vector2i.UP,
			current + Vector2i.DOWN,
			current + Vector2i.LEFT,
			current + Vector2i.RIGHT
		]
		
		for neighbor in neighbors:
			if _is_valid_tile_pos(neighbor) and not visited.has(neighbor):
				queue.append(neighbor)
	
	return reachable

func _draw() -> void:
	queue_redraw()

func get_collectible_positions() -> Array[Vector2i]:
	return collectible_positions

func get_rooms() -> Array[Rect2i]:
	return rooms

func get_random_floor_position() -> Vector2i:
	var floor_positions: Array[Vector2i] = []
	for pos in placed_tiles:
		if placed_tiles[pos] == TileType.FLOOR:
			floor_positions.append(pos)
	if floor_positions.is_empty():
		return Vector2i(-1, -1)
	return floor_positions.pick_random()

func is_walkable(pos: Vector2i) -> bool:
	return placed_tiles.get(pos, TileType.WALL) == TileType.FLOOR

func find_path(from: Vector2i, to: Vector2i) -> PackedVector2Array:
	if not _is_valid_tile_pos(from) or not _is_valid_tile_pos(to):
		return PackedVector2Array()
	if astar_grid.is_point_solid(to):
		return PackedVector2Array()
	
	astar_grid.set_point_solid(from, false)
	var path: PackedVector2Array = astar_grid.get_id_path(from, to)
	astar_grid.set_point_solid(from, true)
	
	var world_path: PackedVector2Array = PackedVector2Array()
	for point in path:
		world_path.append(Vector2(point) * tile_size + tile_size / 2)
	
	return world_path
