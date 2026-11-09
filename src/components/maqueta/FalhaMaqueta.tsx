import { useGLTF } from "@react-three/drei";
import { Component, Fragment, type ReactNode } from "react";

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
    // Fragment com chave (e não um <div style="display: contents">): a ilha Astro client:visible observa o
    // primeiro elemento filho, e um elemento com display: contents nunca entra no ecrã, pelo que a maqueta
    // da página do monumento não chegava a carregar.
    if (!this.state.erro) return <Fragment key={this.state.tentativa}>{this.props.children}</Fragment>;
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
