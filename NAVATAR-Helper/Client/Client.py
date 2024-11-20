import streamlit as st
import socket

st.title("NAVATAR-HELPER")  # Set the title of the Streamlit app

# Initialize session state for messages and input status
if "messages" not in st.session_state:
    st.session_state.messages = []  # Holds the chat messages
if "input_disabled" not in st.session_state:
    st.session_state.input_disabled = False  # Tracks if the input field is disabled
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # Holds the question that has been sent

def send_query_to_server(question):
    """
    Sends a question to the server and retrieves the response.

    Args:
        question (str): The question to send to the server.

    Returns:
        str: The response received from the server, or an error message if communication fails.
    """
    serverName = 'gpu5.cs.oslomet.no'  # Updated server name
    serverPort = 8501  # Port number for the server
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        clientSocket.connect((serverName, serverPort))  # Establish a connection to the server
        
        # Send the question to the server
        clientSocket.sendall(question.encode())
        
        # Receive the response in chunks and accumulate it
        response = ""
        while True:
            part = clientSocket.recv(2048).decode()  # Receive parts of the response
            if not part:
                break
            response += part  # Append received parts to the response string

        return response  # Return the complete response
    except Exception as e:
        return f"Error communicating with server: {str(e)}"  # Return an error message if an exception occurs
    finally:
        clientSocket.close()  # Ensure the socket is closed

def handle_user_input():
    """
    Handles the user input from the chat interface, sends the question to the server, 
    and displays the response received.

    This function manages the display of user messages, waits for the assistant's response, 
    and updates the session state accordingly.
    """
    if st.session_state.pending_question:  # If there is a pending question
        question = st.session_state.pending_question
        st.session_state.pending_question = None  # Clear the pending state

        # Store the user message
        st.session_state.messages.append({"role": "user", "content": question})

        # Display the message in the chat
        with st.chat_message("user"):
            st.write(question)

        # Show a spinner while waiting for the assistant's response
        with st.spinner("Waiting for a response from the assistant..."):
            # Send the question to the server
            response = send_query_to_server(question)

        # Store the assistant's response in the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the response with appropriate line breaks
        with st.chat_message("assistant"):
            st.markdown(response.replace('\n', '  \n'))  # Use markdown for proper line breaks

        # Re-enable the input field once the response is received
        st.session_state.input_disabled = False

        # Rerun to update the UI
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"].replace('\n', '  \n'))  # Ensure correct formatting for assistant messages
        else:
            st.write(message["content"])  # Display user messages

# Input for questions, check if the input field should be disabled
if not st.session_state.input_disabled:
    prompt = st.chat_input(
        "Still assistenten spørsmål om NEET",  # Default message for input
        disabled=st.session_state.input_disabled  # Disable input while waiting for server response
    )

    if prompt:
        # Before sending the request, disable the input field and store the question
        st.session_state.pending_question = prompt
        st.session_state.input_disabled = True

        # Rerun to update the UI after setting query parameters
        st.rerun()

else:
    # While waiting for the response, change the input field text to a waiting message
    st.chat_input("Please wait for a response...", disabled=True)

# If there is a pending question, handle it
if st.session_state.pending_question:
    handle_user_input()  # Process the pending user input
