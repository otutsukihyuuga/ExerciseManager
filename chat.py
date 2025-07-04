from flask import Blueprint, jsonify, session, request
from datetime import datetime
import os
from flask_sock import Sock
import json
from flask_bcrypt import Bcrypt
from kyber_py.kyber import Kyber512
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import base64

chat_bp = Blueprint('chat', __name__)
sock = Sock()
bcrypt = Bcrypt()

# Store MongoDB collection reference
users_collection = None

# WebSocket connection manager
active_connections = {}  # Maps user email to their WebSocket connection

def encrypt_message(message: str, shared_key_hex: str) -> tuple[str, str]:
    """
    Encrypt a message using AES-256-CBC with the shared key.
    Returns (encrypted_message_base64, iv_base64)
    """
    # Convert hex shared key to bytes
    shared_key = bytes.fromhex(shared_key_hex)
    
    # Generate a random IV
    iv = get_random_bytes(AES.block_size)
    
    # Create cipher object and encrypt the data
    cipher = AES.new(shared_key, AES.MODE_CBC, iv)
    
    # Encode and pad the message
    message_bytes = message.encode('utf-8')
    padded_message = pad(message_bytes, AES.block_size)
    
    # Encrypt
    encrypted_message = cipher.encrypt(padded_message)
    
    # Convert to base64 for storage
    encrypted_message_b64 = base64.b64encode(encrypted_message).decode('utf-8')
    iv_b64 = base64.b64encode(iv).decode('utf-8')
    
    return encrypted_message_b64, iv_b64

def decrypt_message(encrypted_message_b64: str, iv_b64: str, shared_key_hex: str) -> str:
    """
    Decrypt a message using AES-256-CBC with the shared key.
    Takes base64 encoded encrypted message and IV, and hex encoded shared key.
    Returns the decrypted message string.
    """
    try:
        # Convert from base64/hex to bytes
        encrypted_message = base64.b64decode(encrypted_message_b64)
        iv = base64.b64decode(iv_b64)
        shared_key = bytes.fromhex(shared_key_hex)
        
        # Create cipher object and decrypt the data
        cipher = AES.new(shared_key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(encrypted_message)
        
        # Unpad the decrypted message
        decrypted = unpad(decrypted_padded, AES.block_size)
        
        # Convert from bytes to string
        return decrypted.decode('utf-8')
    except Exception as e:
        print(f"Error decrypting message: {e}")
        return "[Decryption failed]"

def init_chat(app, mongo_users_collection):
    """Initialize chat module with app and MongoDB collection"""
    global users_collection
    users_collection = mongo_users_collection
    sock.init_app(app)
    app.register_blueprint(chat_bp, url_prefix='/chat')

@chat_bp.route('/create_room', methods=['POST'])
def create_chat_room():
    if 'user' not in session:
        return {'error': 'Unauthorized'}, 401
    
    data = request.get_json()
    recipient_email = data.get('recipient_email')
    
    if not recipient_email:
        return {'error': 'Recipient email is required'}, 400
    
    # Check if recipient exists
    recipient = users_collection.find_one({'email': recipient_email})
    if not recipient:
        return {'error': 'Recipient not found'}, 404
    
    # Check if chat already exists
    current_user = session['user']
    user = users_collection.find_one({'email': current_user})
    existing_chat = next((chat for chat in user.get('chats', []) 
                         if chat['recipient_email'] == recipient_email), None)
    
    if existing_chat:
        return {'success': True}, 200
    
    # Add chat to both users' documents
    chat_data = {
        'recipient_email': recipient_email,
        'messages': []
    }
    
    # Add chat to current user's document
    users_collection.update_one(
        {'email': current_user},
        {'$push': {'chats': chat_data}}
    )
    
    # Add chat to recipient's document
    recipient_chat_data = {
        'recipient_email': current_user,
        'messages': []
    }
    users_collection.update_one(
        {'email': recipient_email},
        {'$push': {'chats': recipient_chat_data}}
    )
    
    # Perform initial handshake after creating the chat room
    handshake_result = perform_handshake()
    if handshake_result[1] != 200:
        return {'success': True, 'warning': 'Chat room created but handshake failed'}, 200
    
    return {'success': True}, 200

@chat_bp.route('/rooms')
def get_chat_rooms():
    if 'user' not in session:
        return {'error': 'Unauthorized'}, 401
    
    user = users_collection.find_one({'email': session['user']})
    chats = user.get('chats', [])
    
    rooms = [{
        'id': chat['recipient_email'],
        'name': chat['recipient_email']
    } for chat in chats]
    
    return jsonify(rooms)

@chat_bp.route('/messages/<recipient_email>')
def get_chat_messages(recipient_email):
    if 'user' not in session:
        return {'error': 'Unauthorized'}, 401
    
    user = users_collection.find_one({'email': session['user']})
    chat = next((chat for chat in user.get('chats', []) 
                 if chat['recipient_email'] == recipient_email), None)
    
    if not chat:
        return {'error': 'Chat not found'}, 404
    
    messages = []
    for msg in chat.get('messages', []):
        try:
            # Get the shared key used for this message
            shared_key = msg.get('shared_key', chat.get('shared_key'))
            
            # If message is encrypted (has IV and shared key), decrypt it
            if msg.get('iv') and shared_key:
                decrypted_content = decrypt_message(msg['content'], msg['iv'], shared_key)
            else:
                decrypted_content = msg['content']
            
            messages.append({
                'content': decrypted_content,
                'sender': msg.get('sender', chat['recipient_email']),
                'timestamp': msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']
            })
        except Exception as e:
            print(f"Error decrypting message: {e}")
            # Include error message for failed decryption
            messages.append({
                'content': '[Decryption failed]',
                'sender': msg.get('sender', chat['recipient_email']),
                'timestamp': msg['timestamp'].isoformat() if isinstance(msg['timestamp'], datetime) else msg['timestamp']
            })
    
    return jsonify(messages)

@chat_bp.route('/handshake', methods=['POST'])
def perform_handshake():
    if 'user' not in session:
        return {'error': 'Unauthorized'}, 401
    
    data = request.get_json()
    recipient_email = data.get('recipient_email')
    
    if not recipient_email:
        return {'error': 'Recipient email is required'}, 400
    
    # Check if recipient exists
    recipient = users_collection.find_one({'email': recipient_email})
    if not recipient:
        return {'error': 'Recipient not found'}, 404
    
    current_user = session['user']
    user = users_collection.find_one({'email': current_user})
    
    # Check if chat exists and already has a shared key
    existing_chat = next((chat for chat in user.get('chats', []) 
                         if chat['recipient_email'] == recipient_email), None)
    
    if existing_chat and existing_chat.get('shared_key'):
        return {'success': True, 'message': 'Shared key already exists'}, 200
    
    # Generate Kyber keys for secure handshake
    pk, sk = Kyber512.keygen()
    shared_key, ciphertext = Kyber512.encaps(pk)
    
    # Convert bytes to hex strings for storage
    shared_key_hex = shared_key.hex()
    
    # Update both users' chat documents with the shared key
    users_collection.update_one(
        {
            'email': current_user,
            'chats.recipient_email': recipient_email
        },
        {'$set': {'chats.$.shared_key': shared_key_hex}}
    )
    
    users_collection.update_one(
        {
            'email': recipient_email,
            'chats.recipient_email': current_user
        },
        {'$set': {'chats.$.shared_key': shared_key_hex}}
    )
    
    return {'success': True, 'message': 'Handshake completed successfully'}, 200

@sock.route('/chat')
def chat_socket(ws):
    if 'user' not in session:
        return
    
    current_user = session['user']
    active_connections[current_user] = ws
    
    try:
        while True:
            data = json.loads(ws.receive())
            recipient_email = data.get('recipient_email')
            content = data.get('content')
            
            if not recipient_email or not content:
                continue
            
            # Find the chat in both users' documents
            user = users_collection.find_one({'email': current_user})
            chat = next((chat for chat in user.get('chats', []) 
                        if chat['recipient_email'] == recipient_email), None)
            
            if not chat:
                continue
            
            # Get the shared key for encryption
            shared_key = chat.get('shared_key')
            if not shared_key:
                continue
            
            # Encrypt the message content
            encrypted_content, iv = encrypt_message(content, shared_key)
            
            # Create message object with encrypted content
            message = {
                'content': encrypted_content,
                'iv': iv,
                'timestamp': datetime.utcnow(),
                'sender': current_user,
                'shared_key': shared_key  # Store the key used for encryption
            }
            
            # Add message to both users' chat threads
            users_collection.update_one(
                {
                    'email': current_user,
                    'chats.recipient_email': recipient_email
                },
                {'$push': {'chats.$.messages': message}}
            )
            
            users_collection.update_one(
                {
                    'email': recipient_email,
                    'chats.recipient_email': current_user
                },
                {'$push': {'chats.$.messages': message}}
            )
            
            try:
                # Decrypt message before sending to clients
                decrypted_content = decrypt_message(encrypted_content, iv, shared_key)
                
                # Prepare message for sending with decrypted content
                message_data = {
                    'recipient_email': recipient_email,
                    'content': decrypted_content,
                    'sender': current_user,
                    'timestamp': message['timestamp'].isoformat()
                }
                
                # Send to sender
                ws.send(json.dumps(message_data))
                
                # Send to recipient if they're connected
                recipient_ws = active_connections.get(recipient_email)
                if recipient_ws:
                    recipient_ws.send(json.dumps(message_data))
            except Exception as e:
                print(f"Error decrypting message for WebSocket: {e}")
                # Send error message if decryption fails
                error_message = {
                    'recipient_email': recipient_email,
                    'content': '[Decryption failed]',
                    'sender': current_user,
                    'timestamp': message['timestamp'].isoformat()
                }
                ws.send(json.dumps(error_message))
                if recipient_ws:
                    recipient_ws.send(json.dumps(error_message))
                
    except Exception as e:
        print(f"Error in chat socket: {e}")
    finally:
        # Clean up connection when done
        if current_user in active_connections:
            del active_connections[current_user] 