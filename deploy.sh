#!/bin/bash

# African Wildlife Classification System Deployment Script
# This script helps deploy the application using Docker

set -e

echo "🦁 African Wildlife Classification System Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_status "Docker and Docker Compose are installed"
}

# Check dataset structure
check_dataset() {
    if [ ! -d "african-wildlife" ]; then
        print_warning "Dataset directory 'african-wildlife' not found"
        print_status "Creating dataset directory structure..."
        mkdir -p african-wildlife/{buffalo,elephant,rhino,zebra}
        print_warning "Please add your images to the respective directories:"
        echo "  - african-wildlife/buffalo/"
        echo "  - african-wildlife/elephant/"
        echo "  - african-wildlife/rhino/"
        echo "  - african-wildlife/zebra/"
        return 1
    fi

    # Check if directories have images
    classes=("buffalo" "elephant" "rhino" "zebra")
    total_images=0

    for class in "${classes[@]}"; do
        count=$(find "african-wildlife/$class" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
        total_images=$((total_images + count))
        if [ $count -eq 0 ]; then
            print_warning "No images found in african-wildlife/$class/"
        else
            print_status "Found $count images in $class directory"
        fi
    done

    if [ $total_images -eq 0 ]; then
        print_error "No images found in dataset directories"
        return 1
    fi

    print_status "Total images found: $total_images"
    return 0
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p saved_models cache reports temp
}

# Build and start the application
deploy_application() {
    print_status "Building and starting the application..."

    # Stop any existing containers
    docker-compose down 2>/dev/null || true

    # Build the image
    print_status "Building Docker image..."
    docker-compose build

    # Start the application
    print_status "Starting the application..."
    docker-compose up -d

    # Wait for the application to start
    print_status "Waiting for application to start..."
    sleep 10

    # Check if the application is running
    if docker-compose ps | grep -q "Up"; then
        print_status "Application is running successfully!"
        echo ""
        echo "🌐 Access the application at: http://localhost:8501"
        echo ""
        echo "📊 To view logs: docker-compose logs -f"
        echo "⏹️  To stop: docker-compose down"
        echo "🔄 To restart: docker-compose restart"
    else
        print_error "Application failed to start"
        echo "Check logs with: docker-compose logs"
        exit 1
    fi
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    echo ""

    # Check prerequisites
    check_docker

    # Check dataset
    if ! check_dataset; then
        read -p "Continue without complete dataset? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Deployment cancelled. Please add images to the dataset."
            exit 1
        fi
    fi

    # Create directories
    create_directories

    # Deploy
    deploy_application

    echo ""
    print_status "Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Navigate to http://localhost:8501 in your browser"
    echo "2. Start with the 'System Overview' to run EDA"
    echo "3. Train models in 'Traditional ML' or 'Deep Learning'"
    echo "4. Use 'Prediction Interface' to classify new images"
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping application..."
        docker-compose down
        ;;
    "restart")
        print_status "Restarting application..."
        docker-compose restart
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        print_warning "This will remove all containers and images"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down
            docker-compose down --rmi all
            print_status "Cleanup completed"
        fi
        ;;
    *)
        main
        ;;
esac