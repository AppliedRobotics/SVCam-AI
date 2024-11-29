gst-launch-1.0 udpsrc port=1234 ! "application/x-rtp,payload=26" ! rtpjpegdepay ! jpegdec ! queue ! autovideosink &
gst-launch-1.0 udpsrc port=1235 ! "application/x-rtp,payload=26" ! rtpjpegdepay ! jpegdec ! queue ! autovideosink &
gst-launch-1.0 udpsrc port=1237 ! "application/x-rtp, payload=26" ! rtpjpegdepay ! jpegdec  ! queue ! autovideosink &

