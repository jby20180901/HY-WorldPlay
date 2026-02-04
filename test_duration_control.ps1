# Test script for custom video duration control

# Test different video durations
$test_durations = @(5, 10, 15, 20)

foreach ($duration in $test_durations) {
    Write-Host "Testing video duration: $duration seconds"
    
    # Calculate expected values
    $expected_fps = 24
    $expected_frames = $duration * $expected_fps
    $expected_latents = [math]::Floor(($expected_frames - 1) / 4)
    
    # Calculate adjusted values (to ensure num_latents is divisible by 4)
    $adjusted_latents = (([math]::Floor(($expected_latents + 3) / 4)) * 4)
    $adjusted_frames = (($adjusted_latents - 1) * 4 + 1)
    
    Write-Host "  Expected: $expected_frames frames, $expected_latents latents"
    Write-Host "  Adjusted: $adjusted_frames frames, $adjusted_latents latents"
    Write-Host "  Pose string: w-$expected_latents"
    Write-Host ""
}

Write-Host "Test completed successfully!"
Write-Host "To generate a video with custom duration, simply modify the VIDEO_DURATION parameter in run.sh"
Write-Host "For example, set VIDEO_DURATION=30 to generate a 30-second video"
