// Draws LocalLab's app icon with Core Graphics and writes a 1024 px PNG.
//
//   swift scripts/make-icon.swift /tmp/icon-1024.png && scripts/make-icon-set.sh /tmp/icon-1024.png
//
// A lab flask (the "Lab") holding a glowing network (AI, running locally), with a sparkle for
// Smart Fit. Drawn on Apple's macOS icon grid: an 824 pt rounded square on a 1024 canvas.
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let size = 1024
let space = CGColorSpace(name: CGColorSpace.displayP3)!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                    space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
// Top-left origin, like a design tool.
ctx.translateBy(x: 0, y: CGFloat(size))
ctx.scaleBy(x: 1, y: -1)

func rgb(_ hex: UInt32, _ a: CGFloat = 1) -> CGColor {
    CGColor(colorSpace: space, components: [
        CGFloat((hex >> 16) & 0xFF) / 255, CGFloat((hex >> 8) & 0xFF) / 255, CGFloat(hex & 0xFF) / 255, a,
    ])!
}
func gradient(_ colors: [CGColor], _ locations: [CGFloat]) -> CGGradient {
    CGGradient(colorsSpace: space, colors: colors as CFArray, locations: locations)!
}

// ── Body: Apple's macOS grid — 824 pt rounded square, centred, soft shadow ──────────────
let body = CGRect(x: 100, y: 100, width: 824, height: 824)
let bodyPath = CGPath(roundedRect: body, cornerWidth: 185, cornerHeight: 185, transform: nil)
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: 12), blur: 28, color: rgb(0x000000, 0.35))
ctx.addPath(bodyPath); ctx.setFillColor(rgb(0x141B4D)); ctx.fillPath()
ctx.restoreGState()

ctx.saveGState()
ctx.addPath(bodyPath); ctx.clip()
ctx.drawLinearGradient(gradient([rgb(0x2B3AA8), rgb(0x1A2168), rgb(0x0B1030)], [0, 0.55, 1]),
                       start: CGPoint(x: 180, y: 100), end: CGPoint(x: 844, y: 924), options: [])
// Soft light from above.
ctx.drawRadialGradient(gradient([rgb(0x8FA2FF, 0.35), rgb(0x8FA2FF, 0)], [0, 1]),
                       startCenter: CGPoint(x: 400, y: 180), startRadius: 0,
                       endCenter: CGPoint(x: 400, y: 180), endRadius: 560, options: [])
// Glow behind the flask's liquid.
ctx.drawRadialGradient(gradient([rgb(0x2EE6C5, 0.30), rgb(0x2EE6C5, 0)], [0, 1]),
                       startCenter: CGPoint(x: 512, y: 700), startRadius: 0,
                       endCenter: CGPoint(x: 512, y: 700), endRadius: 380, options: [])
ctx.restoreGState()

// ── Flask ─────────────────────────────────────────────────────────────────────────────
let neckLeft: CGFloat = 442, neckRight: CGFloat = 582
let neckTop: CGFloat = 262, shoulder: CGFloat = 440
let baseY: CGFloat = 806, baseLeft: CGFloat = 262, baseRight: CGFloat = 762, corner: CGFloat = 70
func wallX(_ y: CGFloat, left: Bool) -> CGFloat {
    let t = (y - shoulder) / (baseY - shoulder)
    return left ? neckLeft + (baseLeft - neckLeft) * t : neckRight + (baseRight - neckRight) * t
}
let flask = CGMutablePath()
flask.move(to: CGPoint(x: neckLeft, y: neckTop))
flask.addLine(to: CGPoint(x: neckLeft, y: shoulder))
flask.addLine(to: CGPoint(x: wallX(baseY - corner, left: true), y: baseY - corner))
flask.addQuadCurve(to: CGPoint(x: baseLeft + corner, y: baseY), control: CGPoint(x: baseLeft - 8, y: baseY))
flask.addLine(to: CGPoint(x: baseRight - corner, y: baseY))
flask.addQuadCurve(to: CGPoint(x: wallX(baseY - corner, left: false), y: baseY - corner),
                   control: CGPoint(x: baseRight + 8, y: baseY))
flask.addLine(to: CGPoint(x: neckRight, y: shoulder))
flask.addLine(to: CGPoint(x: neckRight, y: neckTop))
flask.closeSubpath()

// Glass.
ctx.addPath(flask); ctx.setFillColor(rgb(0xFFFFFF, 0.10)); ctx.fillPath()

// Liquid, clipped to the flask, with a gentle wave on top.
ctx.saveGState()
ctx.addPath(flask); ctx.clip()
let surface: CGFloat = 585
let liquid = CGMutablePath()
liquid.move(to: CGPoint(x: 200, y: surface))
var x: CGFloat = 200
while x <= 824 {
    liquid.addLine(to: CGPoint(x: x, y: surface + 9 * sin((x - 200) / 62)))
    x += 4
}
liquid.addLine(to: CGPoint(x: 824, y: 860)); liquid.addLine(to: CGPoint(x: 200, y: 860)); liquid.closeSubpath()
ctx.addPath(liquid); ctx.clip()
ctx.drawLinearGradient(gradient([rgb(0x49F2D0), rgb(0x1FC7B0), rgb(0x0E8F9A)], [0, 0.5, 1]),
                       start: CGPoint(x: 512, y: surface), end: CGPoint(x: 512, y: baseY), options: [])

// A small network inside the liquid: AI, running right here.
let nodes: [CGPoint] = [
    CGPoint(x: 360, y: 700), CGPoint(x: 450, y: 648), CGPoint(x: 470, y: 750),
    CGPoint(x: 565, y: 680), CGPoint(x: 590, y: 770), CGPoint(x: 668, y: 715),
]
let edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
ctx.setStrokeColor(rgb(0x063B4A, 0.55)); ctx.setLineWidth(9); ctx.setLineCap(.round)
for (a, b) in edges {
    ctx.move(to: nodes[a]); ctx.addLine(to: nodes[b])
}
ctx.strokePath()
for node in nodes {
    ctx.setFillColor(rgb(0x063B4A, 0.75))
    ctx.fillEllipse(in: CGRect(x: node.x - 19, y: node.y - 19, width: 38, height: 38))
    ctx.setFillColor(rgb(0xE9FFFB))
    ctx.fillEllipse(in: CGRect(x: node.x - 11, y: node.y - 11, width: 22, height: 22))
}
ctx.restoreGState()

// Bubbles rising above the surface.
for (cx, cy, r) in [(505.0, 520.0, 16.0), (540.0, 468.0, 11.0), (500.0, 420.0, 8.0)] {
    ctx.setStrokeColor(rgb(0xBFFFF2, 0.9)); ctx.setLineWidth(6)
    ctx.strokeEllipse(in: CGRect(x: cx - r, y: cy - r, width: 2 * r, height: 2 * r))
}

// Glass outline and highlight.
ctx.addPath(flask)
ctx.setStrokeColor(rgb(0xF4F7FF)); ctx.setLineWidth(24); ctx.setLineJoin(.round); ctx.strokePath()
ctx.setStrokeColor(rgb(0xFFFFFF, 0.45)); ctx.setLineWidth(12); ctx.setLineCap(.round)
ctx.move(to: CGPoint(x: 330, y: 700)); ctx.addLine(to: CGPoint(x: 392, y: 590)); ctx.strokePath()

// Rim.
let rim = CGPath(roundedRect: CGRect(x: 414, y: 236, width: 196, height: 44), cornerWidth: 22, cornerHeight: 22, transform: nil)
ctx.addPath(rim); ctx.setFillColor(rgb(0xF4F7FF)); ctx.fillPath()

// Smart Fit sparkle.
func sparkle(_ c: CGPoint, _ r: CGFloat) {
    let p = CGMutablePath()
    p.move(to: CGPoint(x: c.x, y: c.y - r))
    p.addQuadCurve(to: CGPoint(x: c.x + r, y: c.y), control: CGPoint(x: c.x + r * 0.18, y: c.y - r * 0.18))
    p.addQuadCurve(to: CGPoint(x: c.x, y: c.y + r), control: CGPoint(x: c.x + r * 0.18, y: c.y + r * 0.18))
    p.addQuadCurve(to: CGPoint(x: c.x - r, y: c.y), control: CGPoint(x: c.x - r * 0.18, y: c.y + r * 0.18))
    p.addQuadCurve(to: CGPoint(x: c.x, y: c.y - r), control: CGPoint(x: c.x - r * 0.18, y: c.y - r * 0.18))
    ctx.addPath(p); ctx.setFillColor(rgb(0xFFF3B0)); ctx.fillPath()
}
sparkle(CGPoint(x: 712, y: 318), 64)
sparkle(CGPoint(x: 790, y: 420), 28)

let image = ctx.makeImage()!
let out = URL(fileURLWithPath: CommandLine.arguments[1])
let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, image, nil)
CGImageDestinationFinalize(dest)
