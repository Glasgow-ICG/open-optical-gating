//
//  JNSView.m
//  Spim Interface
//
//  Created by Jonathan Taylor on 22/05/2015.
//
//

#import "JNSView.h"

@implementation JNSView

-(void)awakeFromNib
{
    [self addObserver:self forKeyPath:@"viewNeedsRedraw_dummyProperty" options:0 context:NULL];
    [super awakeFromNib];
}

-(void)dealloc
{
    [self removeObserver:self forKeyPath:@"viewNeedsRedraw_dummyProperty"];
    [super dealloc];
}

-(void)observeValueForKeyPath:(NSString *)keyPath
                     ofObject:(id)object
                       change:(NSDictionary *)change
                      context:(void *)context
{
    if ([keyPath isEqualToString:@"viewNeedsRedraw_dummyProperty"])
        self.needsDisplay = true;
    else
        [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
}

-(bool)viewNeedsRedraw_dummyProperty { return true; }

@end
