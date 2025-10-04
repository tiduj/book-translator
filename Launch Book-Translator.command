#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "🚀 Starting Book Translator..."
echo ""

# Kill any existing processes on port 5001
echo "🔪 Killing existing processes on port 5001..."
lsof -ti:5001 | xargs kill -9 2>/dev/null
sleep 1

# Clear cache
echo "🧹 Clearing cache..."
rm -f cache.db translations.db
echo "✓ Cache cleared"
echo ""

# Start the server
echo "🌐 Starting server on http://localhost:5001..."
python3 translator.py &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 3

# Check if server is running
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo "✓ Server started successfully!"
    echo ""
    
    # Open browser
    echo "🌍 Opening browser..."
    open http://localhost:5001
    
    echo ""
    echo "✅ Book Translator is running!"
    echo "📝 Server PID: $SERVER_PID"
    echo "🌐 URL: http://localhost:5001"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Wait for the server process
    wait $SERVER_PID
else
    echo "❌ Failed to start server"
    exit 1
fi
