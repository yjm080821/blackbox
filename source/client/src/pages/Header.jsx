import styles from '../styles/Header.module.css';
import logo from '../assets/logo.png';
import { Mobile, PC } from '../components/responsive'; // 컴포넌트가 있는 파일 경로에 따라 수정 필요

const Header = ({handlePage}) => {
    return (
        <header className={styles.Header}>
        <PC>
            <div className={styles.PC}>
                <img src={logo} className={styles.logo}/>
            </div>
        </PC>
        <Mobile>
            <div className={styles.M}>
                <button className={styles.change} onClick={() => {handlePage(-1, true)}}>{'<<'}</button>
                <img src={logo} className={styles.logo}/>
                <button className={styles.change} onClick={() => {handlePage(1, true)}}>{'>>'}</button>
            </div>
        </Mobile>
        </header>
    );
}

export default Header;