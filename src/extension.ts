// Copyright 2026 laelume. 

import * as vscode from 'vscode';
import { SpectroEditorProvider } from './SpectroPanel';

// === === === === === === === === 
// E X T E N S I O N   E N T R Y P O I N T
// ==== ==== ==== ==== ==== ====

export function activate(context: vscode.ExtensionContext): void {
    // Register the custom editor provider for all supported audio types.
    // Activation is driven entirely by the customEditors selector in package.json;
    // no explicit activationEvents entry is required (VSCode 1.74+).
    context.subscriptions.push(
        SpectroEditorProvider.register(context)
    );
}

export function deactivate(): void {
    // No persistent resources to release.
}