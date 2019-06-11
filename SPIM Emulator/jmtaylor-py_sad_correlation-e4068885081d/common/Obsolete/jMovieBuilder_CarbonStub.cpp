void CopyNSImageToGWorld(const class NSImage *image, GWorldPtr gWorldPtr, const struct _NSRect *cropRect, double gain)
{
	// This function is defined only for Cocoa apps. For Carbon it is easiest to
	// define this function as a stub (which should never be called)
	ALWAYS_ASSERT(0);
}
