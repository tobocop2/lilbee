package structures;

/**
 * A singly linked list of integers with head and tail operations.
 */
public class LinkedList {
    private static class Node {
        int data;
        Node next;
        Node(int data) {
            this.data = data;
            this.next = null;
        }
    }

    private Node head;
    private int size;

    public LinkedList() {
        this.head = null;
        this.size = 0;
    }

    /** Insert a new node at the head of the list. */
    public void insertAtHead(int data) {
        Node newNode = new Node(data);
        newNode.next = head;
        head = newNode;
        size++;
    }

    /** Remove and return the value at the tail of the list. */
    public int removeAtTail() {
        if (head == null) {
            throw new IllegalStateException("List is empty");
        }
        if (head.next == null) {
            int value = head.data;
            head = null;
            size--;
            return value;
        }
        Node current = head;
        while (current.next.next != null) {
            current = current.next;
        }
        int value = current.next.data;
        current.next = null;
        size--;
        return value;
    }
}
