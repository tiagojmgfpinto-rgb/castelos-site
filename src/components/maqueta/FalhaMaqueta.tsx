import { useGLTF } from "@react-three/drei";
import { Component, type ReactNode } from "react";

/**
 * Se a maqueta 3D não carregar (rede em baixo, ficheiro em falta, WebGL perdido), mostra um aviso no lugar
 * dela em vez de deitar abaixo o atlas inteiro; «Tentar outra vez» volta a montá-la.
 */
export class FalhaMaqueta extends Component<{ children: ReactNode; url?: string }, { erro: boolean; tentativa: number }> {
  state = { erro: false, tentativa: 0 };

  static getDerivedStateFromError() {
    return { erro: true };
  }

  componentDidCatch(e: unknown) {
    console.error("[maqueta] não carregou:", e);
  }

  render() {
    if (!this.state.erro) return <div key={this.state.tentativa} style={{ display: "contents" }}>{this.props.children}</div>;
    return (
      <div className="ficha-maqueta-palco maqueta-falha" role="alert">
        <p>Não foi possível carregar a maqueta 3D.</p>
        <button
          type="button"
          onClick={() => {
            // o carregador guarda a falha em cache: limpa-se para a nova tentativa ir mesmo à rede
            if (this.props.url) useGLTF.clear(this.props.url);
            this.setState((s) => ({ erro: false, tentativa: s.tentativa + 1 }));
          }}
        >
          Tentar outra vez
        </button>
      </div>
    );
  }
}
