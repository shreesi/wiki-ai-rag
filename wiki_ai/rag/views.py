from django.shortcuts import render, redirect
from .rag_engine import ask_question


# ---------------------------------------------------------
# Main Home View
# ---------------------------------------------------------

def home(request):
    """
    Handles:
    - Displaying the chat interface
    - Processing user questions
    - Clearing chat history
    - Managing short-term memory via Django sessions
    """

    # -----------------------------------------------------
    # Initialize chat history in session (per user)
    # -----------------------------------------------------
    if "chat" not in request.session:
        # Chat history structure:
        # [
        #   {
        #     "question": "...",
        #     "answer": "...",
        #     "sources": {...}
        #   },
        #   ...
        # ]
        request.session["chat"] = []

    # -----------------------------------------------------
    # Handle POST requests (Ask question / Clear chat)
    # -----------------------------------------------------
    if request.method == "POST":

        # If user clicked "Clear Chat" button
        if "clear_chat" in request.POST:
            request.session["chat"] = []          # Remove all chat history
            request.session.modified = True       # Mark session as updated
            return redirect("/")                  # Refresh page cleanly

        # If user submitted a new question
        question = request.POST.get("question")
        if question:
            # Call RAG pipeline with short-term memory
            # Only last 2 chats will be used internally
            answer, sources = ask_question(
                question,
                chat_history=request.session["chat"]
            )

            # Append new interaction to chat history
            request.session["chat"].append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

            # Ensure Django saves session updates
            request.session.modified = True

    # -----------------------------------------------------
    # Render UI with chat history and memory indicator
    # -----------------------------------------------------
    return render(
        request,
        "index.html",
        {
            "chat": request.session["chat"],

            # Used by UI to show when memory (last 2 chats) is active
            "memory_used": len(request.session["chat"]) >= 2
        }
    )
