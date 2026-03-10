package collections

// Stack is a generic LIFO data structure.
type Stack struct {
	items []interface{}
}

// NewStack returns an initialised empty Stack.
func NewStack() *Stack {
	return &Stack{items: make([]interface{}, 0)}
}

// Push adds an item to the top of the stack.
func (s *Stack) Push(item interface{}) {
	s.items = append(s.items, item)
}

// Pop removes and returns the top item from the stack.
// Returns nil if the stack is empty.
func (s *Stack) Pop() interface{} {
	if len(s.items) == 0 {
		return nil
	}
	top := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return top
}

// Peek returns the top item without removing it.
func (s *Stack) Peek() interface{} {
	if len(s.items) == 0 {
		return nil
	}
	return s.items[len(s.items)-1]
}
