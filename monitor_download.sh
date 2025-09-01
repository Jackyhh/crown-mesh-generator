#!/bin/bash

echo "=== ShapeNet Dataset Download Monitor ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== Download Status ==="
    echo "Timestamp: $(date)"
    echo ""
    
    # Check if download directory exists
    if [ -d "data" ]; then
        echo "📁 Data directory size:"
        du -sh data/ 2>/dev/null || echo "Calculating..."
        echo ""
        
        # Check if shapenet_psr.zip exists
        if [ -f "data/shapenet_psr.zip" ]; then
            echo "📦 Current zip file size:"
            ls -lh data/shapenet_psr.zip | awk '{print $5 " - " $9}'
            echo ""
            echo "🎯 Target size: ~85GB (compressed)"
            echo ""
            
            # Calculate percentage if possible
            CURRENT_SIZE=$(stat -f%z data/shapenet_psr.zip 2>/dev/null || stat -c%s data/shapenet_psr.zip 2>/dev/null || echo "0")
            TARGET_SIZE=91398252796  # ~85GB from the wget output
            if [ "$CURRENT_SIZE" -gt 0 ] && [ "$TARGET_SIZE" -gt 0 ]; then
                PERCENTAGE=$(echo "scale=2; $CURRENT_SIZE * 100 / $TARGET_SIZE" | bc -l 2>/dev/null || echo "Calculating...")
                echo "📊 Progress: $PERCENTAGE%"
            fi
        else
            echo "⏳ Zip file not yet created or still initializing..."
        fi
        
        # Check if extraction has started
        if [ -d "data/shapenet_psr" ]; then
            echo ""
            echo "🔓 Extraction completed! Dataset is ready."
            echo "✅ You can now start training with: ./start_training.sh"
            break
        fi
    else
        echo "📁 Data directory not found - download may not have started"
    fi
    
    echo ""
    echo "⏱️  Next update in 60 seconds..."
    echo "   Use Ctrl+C to stop monitoring"
    
    sleep 60
done