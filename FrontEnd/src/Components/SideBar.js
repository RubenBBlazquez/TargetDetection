import React from 'react';
import {
    CDBSidebar,
    CDBSidebarContent,
    CDBSidebarHeader,
    CDBSidebarMenu,
    CDBSidebarMenuItem,
} from 'cdbreact';
import {NavLink} from 'react-router-dom';
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {icon} from "@fortawesome/fontawesome-svg-core/import.macro";

const Sidebar = () => {
    return (
        <div style={{display: 'flex', height: '100vh', overflow: 'scroll initial'}}>
            <CDBSidebar textColor="#fff" backgroundColor="#333">
                <CDBSidebarHeader prefix={<i className="fa fa-bars fa-large"></i>}>
                    <a href="/" className="text-decoration-none" style={{color: 'inherit'}}>
                        Target Detection
                    </a>
                </CDBSidebarHeader>

                <CDBSidebarContent className="sidebar-content">
                    <CDBSidebarMenu>
                        <NavLink exact to="/predictions" activeClassName="activeClicked">
                            <CDBSidebarMenuItem>
                                <div className={'d-flex'}>
                                    <div className={'me-sm-3'}>
                                        <FontAwesomeIcon className='p-1'
                                                         icon={icon({name: 'chart-line'})}/></div>
                                    <span>Predictions</span>
                                </div>
                                <hr/>
                            </CDBSidebarMenuItem>
                        </NavLink>
                        <NavLink exact to="/training" activeClassName="activeClicked">
                            <CDBSidebarMenuItem>
                                <div className={'d-flex'}>
                                    <div className={'me-sm-3'}>
                                        <FontAwesomeIcon className='p-1'
                                                         icon={icon({name: 'lines-leaning'})}/>
                                    </div>
                                    <span>Training</span>
                                </div>
                                <hr/>
                            </CDBSidebarMenuItem>
                        </NavLink>
                    </CDBSidebarMenu>
                </CDBSidebarContent>
            </CDBSidebar>
        </div>);
};

export default Sidebar;