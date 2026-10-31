#!/bin/bash
#
# Installation script for Bat Activity Monitoring System
# Raspberry Pi Zero 2 W setup
#
# Usage: sudo bash install.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Bat Monitoring System - Installation"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run as root (use sudo)"
    exit 1
fi

# Get the directory where script is located
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installation directory: $INSTALL_DIR"
echo ""

# Raspberry Pi OS Bookworm+ moved config.txt to /boot/firmware
if [ -f /boot/firmware/config.txt ]; then
    BOOT_CONFIG=/boot/firmware/config.txt
else
    BOOT_CONFIG=/boot/config.txt
fi
echo "Boot config: $BOOT_CONFIG"

# Update system
echo "Step 1: Updating system packages..."
apt-get update
apt-get upgrade -y

# Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    i2c-tools \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype-dev \
    liblcms2-dev \
    libopenjp2-7 \
    libtiff6 \
    git

# Enable I2C
echo ""
echo "Step 3: Enabling I2C interface..."
if ! grep -q "^dtparam=i2c_arm=on" "$BOOT_CONFIG"; then
    echo "dtparam=i2c_arm=on" >> "$BOOT_CONFIG"
    echo "I2C enabled (reboot required)"
else
    echo "I2C already enabled"
fi

# Add I2C modules to load at boot
if ! grep -q "^i2c-dev" /etc/modules; then
    echo "i2c-dev" >> /etc/modules
fi

# Enable SPI for TFT display
echo ""
echo "Step 3b: Enabling SPI interface for TFT display..."
if ! grep -q "^dtparam=spi=on" "$BOOT_CONFIG"; then
    echo "dtparam=spi=on" >> "$BOOT_CONFIG"
    echo "SPI enabled (reboot required)"
else
    echo "SPI already enabled"
fi

# Configure I2S audio
echo ""
echo "Step 4: Configuring I2S audio..."
# The INMP441 shows up as a capture device through the googlevoicehat overlay
# (i2s-mmap alone does not create a sound card).
if ! grep -q "^dtparam=i2s=on" "$BOOT_CONFIG"; then
    echo "dtparam=i2s=on" >> "$BOOT_CONFIG"
fi
if ! grep -q "^dtoverlay=googlevoicehat-soundcard" "$BOOT_CONFIG"; then
    echo "dtoverlay=googlevoicehat-soundcard" >> "$BOOT_CONFIG"
    echo "I2S microphone overlay added (reboot required)"
else
    echo "I2S overlay already configured"
fi

# Add user to i2c and audio groups
echo ""
echo "Step 5: Adding user to required groups..."
USER_NAME="${SUDO_USER:-$(logname 2>/dev/null || echo pi)}"
usermod -a -G i2c,audio,gpio,spi,netdev "$USER_NAME"
echo "User $USER_NAME added to i2c, audio, gpio, spi and netdev groups"

# Create virtual environment
echo ""
echo "Step 6: Creating Python virtual environment..."
cd "$INSTALL_DIR"
if [ ! -d "venv" ]; then
    sudo -u "$USER_NAME" python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate venv and install Python packages
echo ""
echo "Step 7: Installing Python packages..."
# Install as the service user so the venv isn't owned by root
PIP=(sudo -u "$USER_NAME" "$INSTALL_DIR/venv/bin/pip")   # array: the path may contain spaces

# Upgrade pip
"${PIP[@]}" install --upgrade pip setuptools wheel

# Install dependencies
"${PIP[@]}" install \
    numpy \
    ai-edge-litert \
    sounddevice \
    soundfile \
    librosa \
    matplotlib \
    pandas \
    scikit-learn \
    smbus2 \
    adafruit-circuitpython-rgb-display \
    Adafruit-Blinka \
    spidev \
    RPi.GPIO \
    Pillow \
    textual

echo "Python packages installed"

# Terminal control panel: `batmon` command, opened automatically on SSH login
echo ""
echo "Step 7b: Installing the batmon control panel..."
cat > /usr/local/bin/batmon << EOF
#!/bin/sh
exec "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/batmon.py" "\$@"
EOF
chmod +x /usr/local/bin/batmon
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
if [ -n "$USER_HOME" ] && ! grep -q ">>> bat-monitor" "$USER_HOME/.bashrc" 2>/dev/null; then
    cat >> "$USER_HOME/.bashrc" << 'EOF'

# >>> bat-monitor: open the control panel when you SSH in (q = back to the shell) >>>
# Turn off with: batmon --autostart off
if [ -n "$SSH_TTY" ] && [ -z "$TMUX" ] && [ -z "$BATMON_NO_AUTOSTART" ]    && [ ! -e "$HOME/.batmon-no-autostart" ] && command -v batmon >/dev/null 2>&1; then
    BATMON_NO_AUTOSTART=1 batmon
fi
# <<< bat-monitor <<<
EOF
    chown "$USER_NAME": "$USER_HOME/.bashrc"
fi
echo "batmon installed (opens automatically on SSH login)"

# Create systemd service
echo ""
echo "Step 8: Installing systemd service..."
cat > /etc/systemd/system/bat-monitor.service << EOF
[Unit]
Description=Bat Activity Monitoring System
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart="$INSTALL_DIR/venv/bin/python3" "$INSTALL_DIR/main.py"
Restart=on-failure
RestartSec=10
# Time to finish tonight's report email when stopped
TimeoutStopSec=120
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload
echo "Systemd service installed"

# Create test script
echo ""
echo "Step 9: Creating hardware test script..."
cat > "$INSTALL_DIR/test_hardware.py" << 'EOF'
#!/usr/bin/env python3
"""Quick hardware test for bat monitoring system."""

import sys

def test_i2c():
    """Test I2C devices."""
    print("\n=== I2C Devices ===")
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        
        # Scan for devices
        devices = []
        for addr in range(0x03, 0x78):
            try:
                bus.read_byte(addr)
                devices.append(hex(addr))
            except:
                pass
        
        if devices:
            print(f"Found devices: {', '.join(devices)}")
            if '0x68' in devices:
                print("  ✓ DS3231 RTC detected (0x68)")
        else:
            print("  ✗ No I2C devices found")
        
        bus.close()
        return len(devices) > 0
    except Exception as e:
        print(f"  ✗ I2C test failed: {e}")
        return False

def test_spi():
    """Test SPI interface for TFT display."""
    print("\n=== SPI Interface (TFT Display) ===")
    try:
        import os
        spi_devices = [f for f in os.listdir('/dev') if f.startswith('spidev')]
        if spi_devices:
            print(f"  ✓ SPI devices found: {', '.join(spi_devices)}")
            return True
        else:
            print("  ✗ No SPI devices found. Enable SPI in raspi-config.")
            return False
    except Exception as e:
        print(f"  ✗ SPI test failed: {e}")
        return False

def test_display():
    """Test TFT display initialization."""
    print("\n=== TFT Display (ST7789) ===")
    try:
        import digitalio
        import board
        from adafruit_rgb_display import st7789
        
        # Try to initialize display
        cs_pin = digitalio.DigitalInOut(board.CE0)
        dc_pin = digitalio.DigitalInOut(board.D24)
        reset_pin = digitalio.DigitalInOut(board.D25)
        spi = board.SPI()
        
        display = st7789.ST7789(
            spi,
            cs=cs_pin,
            dc=dc_pin,
            rst=reset_pin,
            width=240,
            height=320,
            rotation=90
        )
        
        print("  ✓ TFT display initialized (240x320, ST7789)")
        return True
    except Exception as e:
        print(f"  ✗ Display test failed: {e}")
        print("     Check wiring and SPI enablement")
        return False

def test_audio():
    """Test audio input."""
    print("\n=== Audio Devices ===")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("Available devices:")
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{idx}] {dev['name']} (input)")
        return True
    except Exception as e:
        print(f"  ✗ Audio test failed: {e}")
        return False

def test_gpio():
    """Test GPIO."""
    print("\n=== GPIO ===")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        print("  ✓ GPIO initialized")
        GPIO.cleanup()
        return True
    except Exception as e:
        print(f"  ✗ GPIO test failed: {e}")
        return False

def test_model():
    """Test model loading."""
    print("\n=== AI Model ===")
    try:
        from model_inference import BatClassifier
        classifier = BatClassifier("12_29_both_species.tflite", "label_encoder.pkl")
        print(f"  ✓ Model loaded ({len(classifier.class_names)} classes)")
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Bat Monitoring System - Hardware Test")
    print("=" * 50)
    
    results = {
        "I2C": test_i2c(),
        "SPI": test_spi(),
        "TFT Display": test_display(),
        "Audio": test_audio(),
        "GPIO": test_gpio(),
        "Model": test_model()
    }
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:15s}: {status}")
    
    print("\n")
    
    if all(results.values()):
        print("✓ All tests passed! System ready.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Check hardware connections.")
        sys.exit(1)
EOF

chmod +x "$INSTALL_DIR/test_hardware.py"
echo "Hardware test script created"

# Summary
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Reboot the Raspberry Pi to enable I2C, SPI, and I2S:"
echo "   sudo reboot"
echo ""
echo "2. After reboot, test hardware:"
echo "   cd \"$INSTALL_DIR\""
echo "   source venv/bin/activate"
echo "   python test_hardware.py"
echo ""
echo "3. Control panel: SSH in and it opens automatically, or type: batmon"
echo "   (q quits to the normal shell; turn auto-open off with: batmon --autostart off)"
echo ""
echo "4. Test the TFT display interface:"
echo "   python ui_controller.py"
echo ""
echo "5. Enable auto-start (optional):"
echo "   sudo systemctl enable bat-monitor.service"
echo "   sudo systemctl start bat-monitor.service"
echo ""
echo "6. View logs:"
echo "   journalctl -u bat-monitor.service -f"
echo ""
echo "Hardware: 2.0\" TFT LCD with EC11 Encoder (ST7789)"
echo "=========================================="
