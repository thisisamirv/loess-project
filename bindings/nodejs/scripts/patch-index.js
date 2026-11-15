'use strict'

// napi-rs regenerates the loader and declarations on every build. Restore the
// async result type, status union, and Symbol.dispose hooks after generation.

const fs = require('fs')
const path = require('path')

const indexPath = path.join(__dirname, '..', 'index.js')
const declarationsPath = path.join(__dirname, '..', 'index.d.ts')
const disposeMarker = '// fastloess Symbol.dispose hooks'

const disposeHooks = `
${disposeMarker}
function __fastloessAttachDispose(ctor) {
  if (!ctor || !ctor.prototype || typeof ctor.prototype.close !== 'function') {
    return
  }
  if (typeof Symbol !== 'undefined' && Symbol.dispose && !ctor.prototype[Symbol.dispose]) {
    ctor.prototype[Symbol.dispose] = function () {
      this.close()
    }
  }
}
__fastloessAttachDispose(module.exports.OnlineLoess)
__fastloessAttachDispose(module.exports.StreamingLoess)
__fastloessAttachDispose(module.exports.LoessResult)
`

let contents = fs.readFileSync(indexPath, 'utf8').replace(/\r\n/g, '\n')
contents = contents.replace(
  /bindingPackageVersion !== '\d+\.\d+\.\d+'/g,
  "bindingPackageVersion !== require('./package.json').version"
)
contents = contents.replace(
  /expected \d+\.\d+\.\d+ but got/g,
  () => "expected ${require('./package.json').version} but got"
)
if (!contents.includes(disposeMarker)) {
  contents += disposeHooks
}
fs.writeFileSync(indexPath, contents)

let declarations = fs.readFileSync(declarationsPath, 'utf8').replace(/\r\n/g, '\n')
const asyncResultType = /(fit_async\([\s\S]*?\): Promise<)unknown(>)/
declarations = declarations.replace(asyncResultType, '$1LoessResult$2')
if (!/fit_async\([\s\S]*?\): Promise<LoessResult>/.test(declarations)) {
  throw new Error('Could not locate the generated fit_async declaration')
}
declarations = declarations.replace(
  /(export interface AddPointResult \{\n\s+\/\*\*[\s\S]*?\*\/\n\s+status: )string(\n)/,
  '$1\'filling\' | \'ready\' | \'dropped\'$2'
)
if (!declarations.includes("status: 'filling' | 'ready' | 'dropped'")) {
  // Fallback if napi omits the doc comment block.
  declarations = declarations.replace(
    /(export interface AddPointResult \{\n\s+status: )string(\n)/,
    '$1\'filling\' | \'ready\' | \'dropped\'$2'
  )
}
fs.writeFileSync(declarationsPath, declarations)
